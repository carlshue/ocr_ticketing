import logging
import re
import numpy as np
import cv2
import asyncio
import psutil
import torch
import gc
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_ocr_and_models(lang='es'):
    """
    Ensures that PaddleOCR is installed and that its models are downloaded.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.info("PaddleOCR not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "paddleocr"])
        from paddleocr import PaddleOCR

    # PaddleOCR manages its own model download automatically
    logger.info(f"Initializing PaddleOCR with language: {lang}")

    # The lang argument in PaddleOCR uses a different notation (e.g., 'es' or 'es+en')
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    
    # Dummy run to ensure models are downloaded
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img_path = "/tmp/dummy_image.jpg"
    cv2.imwrite(dummy_img_path, dummy_img)
    _ = ocr.ocr(dummy_img_path, cls=True)

    logger.info("PaddleOCR models downloaded and initialized.")
    return ocr


def read_imagefile(file_bytes: bytes) -> np.ndarray:
    image_data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img


async def run_ocr_with_timeout(reader, img: np.ndarray, timeout: int = 15):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, reader.ocr, img, True, True, True),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError("OCR processing timed out.")


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


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


def log_system_resources(prefix=""):
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    logger.info(f"{prefix}CPU: {cpu_percent:.2f}%, RAM: {mem.percent:.2f}% "
                f"({mem.used / (1024 ** 2):.2f} MB / {mem.total / (1024 ** 2):.2f} MB)")


def process_ocr_results(results):
    original_texts, cleaned_texts, confidences, bboxes = [], [], [], []

    for line in results:
        if isinstance(line, list):
            for det in line:
                if len(det) != 2:
                    continue
                bbox, (text, conf) = det
                cleaned, _ = desleet_text(clean_text(text))
                original_texts.append(text)
                cleaned_texts.append(cleaned)
                confidences.append(round(conf, 2))
                bboxes.append(bbox)

    return {
        "original_texts": [convert_numpy_types(x) for x in original_texts],
        "cleaned_texts": [convert_numpy_types(x) for x in cleaned_texts],
        "confidences": [convert_numpy_types(x) for x in confidences],
        "bboxes": [convert_numpy_types(x) for x in bboxes],
    }



def cleanup_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
