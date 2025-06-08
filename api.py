"""
OCR API with FastAPI, EasyOCR, and system resource logging.

This API receives an image, performs OCR, and returns the extracted text.
Includes error handling, timeouts, and logs CPU/RAM usage for easier debugging.
"""

import asyncio
import logging
import os
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import easyocr
import numpy as np
import cv2
import psutil  # for system resource usage

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# FastAPI app initialization
# ----------------------------------------------------------
app = FastAPI()

# ----------------------------------------------------------
# EasyOCR reader initialization and warm-up
# ----------------------------------------------------------
logger.info("Initializing EasyOCR reader...")
reader = easyocr.Reader(['es'], gpu=False)  # Disable GPU to avoid server errors
# Warm-up with a dummy image to download weights if needed
dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
reader.readtext(dummy_img)
logger.info("EasyOCR reader ready.")

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------
def desleet_text(text: str) -> tuple[str, int]:
    """
    Convert leet speak text to regular text.

    Returns:
        A tuple of (cleaned text, number of modifications)
    """
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

    # Divide text into tokens (words, spaces)
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
        # Too many modifications? Probably false positive.
        return text.upper(), 0
    else:
        return resultado, modificaciones_leet


def clean_text(text: str) -> str:
    """
    Remove unwanted characters and normalize OCR text.
    """
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
    """
    Convert uploaded file bytes into a NumPy image array using OpenCV.
    """
    image_data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img


async def run_ocr_with_timeout(img: np.ndarray, timeout: int = 10):
    """
    Run OCR with a timeout to prevent hanging requests.

    Args:
        img (np.ndarray): Input image.
        timeout (int): Timeout in seconds.

    Returns:
        List of OCR results.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, reader.readtext, img),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError("OCR processing timed out.")


def log_system_resources():
    """
    Logs current CPU and RAM usage.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    logger.info(f"CPU usage: {cpu_percent:.2f}%, RAM usage: {mem.percent:.2f}% "
                f"({mem.used / (1024 ** 2):.2f} MB / {mem.total / (1024 ** 2):.2f} MB)")


# ----------------------------------------------------------
# API Endpoint
# ----------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Receive an uploaded image, perform OCR, and return the extracted text.

    Steps:
    1. Read the file and decode it as an image.
    2. Run OCR with a timeout.
    3. Clean the text using custom functions.
    4. Return JSON with original, cleaned texts and confidence levels.
    5. Log system resources for debugging.
    """
    try:
        logger.info("Received OCR request.")
        log_system_resources()

        contents = await file.read()
        img = read_imagefile(contents)
        logger.info(f"Image shape: {img.shape}")

        results = await run_ocr_with_timeout(img, timeout=10)
        logger.info(f"OCR completed with {len(results)} results.")

        original_texts = []
        cleaned_texts = []
        confidences = []

        for bbox, text, conf in results:
            cleaned, _ = desleet_text(clean_text(text))
            original_texts.append(text)
            cleaned_texts.append(cleaned)
            confidences.append(round(conf, 2))

        response_data = {
            "original_texts": original_texts,
            "cleaned_texts": cleaned_texts,
            "confidences": confidences
        }

        return JSONResponse(content=response_data)

    except TimeoutError as e:
        logger.error("OCR request timed out.")
        return JSONResponse(content={"error": str(e)}, status_code=504)

    except Exception as e:
        logger.exception("Unexpected error in /ocr endpoint.")
        return JSONResponse(content={"error": str(e)}, status_code=500)
