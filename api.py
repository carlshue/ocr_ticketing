from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
import torch
from fastapi.responses import FileResponse
import os
from pathlib import Path

from ocr_utils import *

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
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
# PaddleOCR reader initialization
# ----------------------------------------------------------
logger.info("PaddleOCR is currently down...")
reader = None#ensure_ocr_and_models(lang='es')
logger.info("PaddleOCR is currently down...")


# ----------------------------------------------------------
# OCR Endpoint
# ----------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(request: Request, file: UploadFile = File(...)):
    
    print("not currently supported PaddleOCR is currently down...")
    return JSONResponse(content={"PaddleOCR OCR SERVERSIDE IS DOWN."}, status_code=500)

    
    request_id = str(uuid.uuid4())
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"[{request_id}] Nueva petición OCR desde {client_host}, User-Agent: {user_agent}")

    try:
        log_system_resources(prefix=f"[{request_id}] Antes OCR: ")

        contents = await file.read()
        img = read_imagefile(contents)
        logger.info(f"[{request_id}] Imagen recibida con shape {img.shape}")

        raw_results = await run_ocr_with_timeout(reader, img, timeout=15)
        logger.info(f"[{request_id}] OCR terminado con {len(raw_results[0]) if raw_results and raw_results[0] else 0} resultados")

        processed = process_ocr_results(raw_results)

        # Corrección de skew
        img_corrected, processed_corrected = rotate_image_and_boxes(img, processed["bboxes"],
                                                                   estimate_skew_angle_from_ocr(processed["bboxes"]))
        processed["bboxes"] = processed_corrected

        # Centros
        centers = get_centers(processed["bboxes"])
        if len(centers) == 0:
            raise RuntimeError("No se detectaron centros en las cajas OCR")

        # Clusterizado
        labels = cluster_centers(centers, eps=50, min_samples=1)

        # Imagen para dibujar conexiones (opcional)
        img_debug = img_corrected.copy()

        # Conexiones
        connected = connect_clusters_lines(img_debug, centers, labels)

        # Filas conectadas
        connected_rows = build_connected_rows(connected)

        # Construcción tabla DataFrame
        df = build_table(processed, labels, connected_rows)

        response_data = {
            "request_id": request_id,
            "table": df.fillna("").values.tolist(),
            "columns": df.columns.tolist() if df.columns is not None else [],
            "ocr_raw": processed,  # puedes eliminarlo si no quieres enviar el raw
        }

        logger.info(f"[{request_id}] Preparando respuesta.")

        cleanup_memory(img, raw_results, contents)
        torch.cuda.empty_cache()
        log_system_resources(prefix=f"[{request_id}] Después limpieza: ")
        logger.info(f"[{request_id}] Petición finalizada.")

        return JSONResponse(content=response_data)

    except TimeoutError as e:
        logger.error(f"[{request_id}] Timeout en OCR")
        return JSONResponse(content={"error": str(e), "request_id": request_id}, status_code=504)

    except Exception as e:
        logger.exception(f"[{request_id}] Error inesperado")
        return JSONResponse(content={"error": str(e), "request_id": request_id}, status_code=500)
    
    
    
# ----------------------------------------------------------
# Nuevo Endpoint: /ocr-json
'''

desleet_text
get_centers
cluster_centers
connect_clusters_lines
build_connected_rows
build_table
'''


# ----------------------------------------------------------
@app.post("/ocr-json")
async def ocr_json_endpoint(request: Request):
    try:
        payload = await request.json()
        logger.info("Recibido JSON en /ocr-json")
        elements = payload.get("elements", [])

        original_texts = []
        cleaned_texts = []
        bboxes = []
        for el in elements:
            text = el.get("text", "")
            cleaned, _ = desleet_text(clean_text(text))
            cleaned_texts.append(cleaned)
            original_texts.append(text)

            left = el["boundingBox"]["left"]
            top = el["boundingBox"]["top"]
            right = el["boundingBox"]["right"]
            bottom = el["boundingBox"]["bottom"]

            bbox = [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom]
            ]
            bboxes.append(bbox)

        processed = {
            "original_texts": original_texts,
            "cleaned_texts": cleaned_texts,
            "confidences": [1.0] * len(original_texts),
            "bboxes": bboxes,
        }

        #OLD
        #centers = get_centers(bboxes)
        #labels = cluster_centers(centers, eps=50, min_samples=1)
        #
        # New:
        
        centers = get_centers(bboxes)
        labels = vertical_aligned_clustering(bboxes)

        # Asumiendo connect_clusters_lines y build_connected_rows están definidas
        #connected = connect_clusters_lines(np.zeros((1, 1, 3), dtype=np.uint8), centers, labels)
        #connected_rows = build_connected_rows(connected)
        connected_rows = build_rows_from_centers(centers,bboxes)
        df = build_table(processed, labels, connected_rows)

        logger.info("Tabla construida:\n", df.to_string(index=False))
        print("Tabla construida:\n", df.to_string(index=False))


        return JSONResponse(content={"status": "ok", "rows": df.to_dict(orient="records")})

    except Exception as e:
        logger.exception("Error procesando /ocr-json")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = os.path.join(BASE_DIR, "static", "index.html")

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    return FileResponse(INDEX_PATH)