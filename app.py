import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pymongo import MongoClient

from main import _setup_logging, run_pipeline


APP_TEMP_ROOT = Path("temp")
UPLOAD_ROOT = APP_TEMP_ROOT / "api_uploads"
LOG_ROOT = APP_TEMP_ROOT / "api_logs"
RESULT_ROOT = APP_TEMP_ROOT / "api_results"

app = FastAPI(title="PDF Infographic Extraction API", version="1.0.0")


def _get_mongo_collection():

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://admin:admin123@127.0.0.1:27017/?authSource=admin")
    mongo_db = os.getenv("MONGODB_DB", "pdf_infographic_extractor")
    mongo_collection = os.getenv("MONGODB_COLLECTION", "final_results")

    client = MongoClient(mongo_uri)
    return client[mongo_db][mongo_collection]


def _build_request_signature(file_hashes):

    normalized_hashes = sorted(file_hashes)
    signature_payload = json.dumps(normalized_hashes, separators=(",", ":"))
    return hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()


def _save_uploaded_files(files: List[UploadFile]):

    batch_id = uuid.uuid4().hex
    batch_dir = UPLOAD_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    file_hashes = []
    file_names = []
    for upload in files:
        if not upload.filename:
            continue
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {upload.filename}")

        file_bytes = upload.file.read()
        destination = batch_dir / Path(upload.filename).name
        with destination.open("wb") as file:
            file.write(file_bytes)
        saved_files.append(destination)
        file_names.append(destination.name)
        file_hashes.append(hashlib.sha256(file_bytes).hexdigest())

    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid PDF files were uploaded.")

    request_signature = _build_request_signature(file_hashes)
    return batch_id, batch_dir, saved_files, request_signature, file_names, file_hashes


def _build_cached_preview(final_payload):

    results = final_payload.get("results", []) if isinstance(final_payload, dict) else []
    preview = []
    for result in results:
        preview.append(
            {
                "file_name": result.get("file_name"),
                "file_path": result.get("file_path"),
                "image_page_count": len(result.get("image_pages", [])),
                "infographic_page_count": len(result.get("infographic_results", [])),
            }
        )
    return preview


@app.get("/api/health")
def health():

    return {"status": "ok"}


@app.post("/api/extract")
async def extract_pdfs(
    files: List[UploadFile] = File(...),
    max_workers: int = Form(4),
    config_path: str = Form("config.json"),
):

    batch_id, batch_dir, saved_files, request_signature, file_names, file_hashes = _save_uploaded_files(files)
    log_file = LOG_ROOT / f"{batch_id}.log"
    output_file = RESULT_ROOT / f"{batch_id}.json"

    _setup_logging(str(log_file))

    try:
        collection = _get_mongo_collection()
        cached_document = await run_in_threadpool(
            collection.find_one,
            {"request_signature": request_signature},
            {"_id": 0, "final_payload": 1},
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"MongoDB lookup failed: {error}") from error

    if cached_document and cached_document.get("final_payload"):
        return cached_document["final_payload"]

    try:
        final_payload = await run_in_threadpool(
            run_pipeline,
            str(batch_dir),
            str(APP_TEMP_ROOT),
            config_path,
            str(output_file),
            max_workers,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    document = {
        "request_signature": request_signature,
        "file_names": sorted(file_names),
        "file_hashes": sorted(file_hashes),
        "file_count": len(saved_files),
        "config_path": config_path,
        "max_workers": max_workers,
        "final_payload": final_payload,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        await run_in_threadpool(
            collection.replace_one,
            {"request_signature": request_signature},
            document,
            True,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"MongoDB save failed: {error}") from error

    return final_payload


@app.get("/api/extract/all")
async def get_cached_extractions():

    try:
        collection = _get_mongo_collection()
        cached_documents = await run_in_threadpool(
            lambda: list(
                collection.find(
                    {},
                    {
                        "_id": 0,
                        "request_signature": 1,
                        "file_names": 1,
                        "file_count": 1,
                        "created_at": 1,
                        "final_payload": 1,
                    },
                ).sort("created_at", -1)
            )
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"MongoDB lookup failed: {error}") from error

    return {
        "cached_results": [
            {
                "request_signature": document.get("request_signature"),
                "file_names": document.get("file_names", []),
                "file_count": document.get("file_count", 0),
                "created_at": document.get("created_at"),
                "preview": _build_cached_preview(document.get("final_payload", {})),
            }
            for document in cached_documents
        ]
    }


@app.get("/api/result/{filename}")
async def get_result_by_filename(filename: str):

    try:
        collection = _get_mongo_collection()
        document = await run_in_threadpool(
            collection.find_one,
            {"file_names": filename},
            {"_id": 0},
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"MongoDB lookup failed: {error}") from error

    if not document:
        raise HTTPException(status_code=404, detail=f"No cached result found for filename: {filename}")

    return document
