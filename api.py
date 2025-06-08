"""
OCR API with FastAPI, EasyOCR, and system resource logging.

This API receives an image, performs OCR, and returns the extracted text.
Includes error handling, timeouts, memory cleanup, and logs CPU/RAM usage and client info for easier debugging.
"""

import asyncio
import logging
import re
import os
import gc
import uuid
from datetime import datetime
from pathlib import Path
import json

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Required libraries
import numpy as np
import cv2
import psutil
import torch

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# FastAPI app initialization
# ----------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# EasyOCR reader initialization and warm-up
# ----------------------------------------------------------
def ensure_easyocr_and_models(lang='es'):
    """
    Ensures that EasyOCR is installed and that its models are downloaded.
    If missing, downloads them and initializes the reader.
    """
    try:
        import easyocr
    except ImportError:
        logger.info("EasyOCR not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "easyocr"])
        import easyocr

    # Check model path
    home_dir = Path.home()
    model_dir = home_dir / ".EasyOCR" / "model"
    detection_model = model_dir / "detection" / "craft_mlt_25k.pth"
    recognition_model = model_dir / "recognition" / f"{lang}.pth"

    if not detection_model.exists() or not recognition_model.exists():
        logger.info("EasyOCR models not found locally. Downloading them...")
        reader = easyocr.Reader([lang], gpu=False)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        reader.readtext(dummy_img)  # Forces download
        logger.info("EasyOCR models downloaded successfully.")
    else:
        logger.info("EasyOCR models already present. Skipping download.")

    # Return a reader instance
    return easyocr.Reader([lang], gpu=False)


logger.info("Initializing EasyOCR reader...")
reader = ensure_easyocr_and_models(lang='es')
logger.info("EasyOCR reader ready.")

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------
def desleet_text(text: str) -> tuple[str, int]:
    leet_dict = {
        '4': 'A', '@': 'A',
        '8': 'B',
        '(': 'C', '<': 'C', '{': 'C', '[': 'C',
        '3': 'E',
        '6': 'G',
        '1': 'I', '!': 'I', '|': 'I',
        '0': 'O',
        '5': 'S', '$': 'S',
        '7': 'T',
        '2': 'Z'
    }

    modificaciones_leet = 0
    tokens = re.findall(r'\S+|\s+', text)
    resultado = ""

    for token in tokens:
        if token.isspace():
            resultado += token
            continue

        if re.fullmatch(r'[\d.,]+', token):
            resultado += token
        else:
            nuevo_token = ""
            for c in token:
                c_upper = c.upper()
                if c_upper in leet_dict:
                    nuevo_token += leet_dict[c_upper]
                    modificaciones_leet += 1
                else:
                    nuevo_token += c_upper
            resultado += nuevo_token

    texto_sin_espacios = re.sub(r'\s+', '', text)
    longitud = len(texto_sin_espacios)

    if longitud > 0 and modificaciones_leet >= (longitud / 2):
        return text.upper(), 0
    else:
        return resultado, modificaciones_leet


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[^A-Za-z0-9ñÑáéíóúÁÉÍÓÚ.,€$ \-]', '', text)
    replacements = {
        'O': '0', 'l': '1', 'I': '1', 'Z': '2',
        'S': '5', 'B': '8'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def read_imagefile(file_bytes: bytes) -> np.ndarray:
    image_data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img


async def run_ocr_with_timeout(img: np.ndarray, timeout: int = 15):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, reader.readtext, img),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError("OCR processing timed out.")


def log_system_resources(prefix=""):
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    logger.info(f"{prefix}CPU usage: {cpu_percent:.2f}%, RAM usage: {mem.percent:.2f}% "
                f"({mem.used / (1024 ** 2):.2f} MB / {mem.total / (1024 ** 2):.2f} MB)")


# ----------------------------------------------------------
# API Endpoint
# ----------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Receive an uploaded image, perform OCR, and return the extracted text.
    Logs client info, system resources, and cleans memory after processing.
    """
    request_id = str(uuid.uuid4())
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"[{request_id}] Received OCR request from {client_host} with User-Agent: {user_agent}")

    try:
        log_system_resources(prefix=f"[{request_id}] Before OCR: ")

        contents = await file.read()
        img = read_imagefile(contents)
        logger.info(f"[{request_id}] Image shape: {img.shape}")

        results = await run_ocr_with_timeout(img, timeout=15)
        logger.info(f"[{request_id}] OCR completed with {len(results)} results.")

        original_texts = []
        cleaned_texts = []
        confidences = []
        bboxes = []

        for bbox, text, conf in results:
            cleaned, _ = desleet_text(clean_text(text))
            original_texts.append(text)
            cleaned_texts.append(cleaned)
            confidences.append(round(conf, 2))
            bboxes.append(bbox)  # Añadimos la caja

        response_data = {
            "request_id": request_id,
            "original_texts": original_texts,
            "cleaned_texts": cleaned_texts,
            "confidences": confidences,
            "bboxes": bboxes
        }

        # Memory cleanup
        del img
        del results
        del contents
        gc.collect()
        torch.cuda.empty_cache()

        log_system_resources(prefix=f"[{request_id}] After cleanup: ")

        logger.info(f"[{request_id}] Finished processing request.")

        return JSONResponse(content=response_data, dumps=lambda obj: json.dumps(obj, default=str))

    except TimeoutError as e:
        logger.error(f"[{request_id}] OCR request timed out.")
        return JSONResponse(content={"error": str(e), "request_id": request_id}, status_code=504)

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error in /ocr endpoint.")
        return JSONResponse(content={"error": str(e), "request_id": request_id}, status_code=500)
