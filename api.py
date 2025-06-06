from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
import cv2
import logging
from utils_ocr import clean_text, desleet_text


#cmon man work
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# Inicializa EasyOCR una vez
reader = easyocr.Reader(['es'])

# Función para convertir una imagen subida en un array de OpenCV
def read_imagefile(file_bytes: bytes) -> np.ndarray:
    image_data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img

# Endpoint principal
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = read_imagefile(contents)
        results = reader.readtext(img)

        original_texts = []
        cleaned_texts = []
        confidences = []

        for bbox, text, conf in results:
            cleaned = desleet_text(clean_text(text))
            original_texts.append(text)
            cleaned_texts.append(cleaned)
            confidences.append(round(conf, 2))

        return JSONResponse(content={
            "original_texts": original_texts,
            "cleaned_texts": cleaned_texts,
            "confidences": confidences
        })

    except Exception as e:
        logger.exception("Error en el endpoint /ocr")
        return JSONResponse(content={"error": str(e)}, status_code=500)