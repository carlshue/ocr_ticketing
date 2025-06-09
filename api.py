from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import logging
import asyncio

import torch

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
logger.info("Initializing PaddleOCR reader...")
reader = ensure_ocr_and_models(lang='es')
logger.info("PaddleOCR reader ready.")

# ----------------------------------------------------------
# OCR Endpoint
# ----------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(request: Request, file: UploadFile = File(...)):
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