from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import logging
import asyncio

import torch

from ocr_utils import (
    ensure_ocr_and_models,
    read_imagefile,
    run_ocr_with_timeout,
    process_ocr_results,
    log_system_resources,
    cleanup_memory
)

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
    logger.info(f"[{request_id}] Received OCR request from {client_host} with User-Agent: {user_agent}")

    try:
        log_system_resources(prefix=f"[{request_id}] Before OCR: ")

        contents = await file.read()
        img = read_imagefile(contents)
        logger.info(f"[{request_id}] Image shape: {img.shape}")

        results = await run_ocr_with_timeout(reader, img, timeout=15)
        logger.info(f"[{request_id}] OCR completed with {len(results[0]) if results and results[0] else 0} results.")

        response_data = process_ocr_results(results)
        response_data["request_id"] = request_id

        logger.info(f"[{request_id}] Response data prepared: {json.dumps(response_data, ensure_ascii=False)}")

        # Memory cleanup
        cleanup_memory(img, results, contents)
        torch.cuda.empty_cache()

        log_system_resources(prefix=f"[{request_id}] After cleanup: ")

        logger.info(f"[{request_id}] Finished processing request.")

        return JSONResponse(content=response_data)

    except TimeoutError as e:
        logger.error(f"[{request_id}] OCR request timed out.")
        return JSONResponse(content={"error": str(e), "request_id": request_id}, status_code=504)

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error in /ocr endpoint.")
        return JSONResponse(content={"error": str(e), "request_id": request_id}, status_code=500)
