import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, Depends
from starlette.status import HTTP_202_ACCEPTED

from src.ingestion.indexer import IngestionService
from src.api.v1.dependencies import get_ingestion_service

router = APIRouter()


async def run_ingestion(service: IngestionService, tmp_dir: str, kb_id: str):
    """Background task to run ingestion and cleanup."""
    tmp_path = Path(tmp_dir)
    try:
        for file_path in tmp_path.iterdir():
            if file_path.is_file():
                await service.ingest(file_path, kb_id)
    finally:
        # Cleanup temporary files
        shutil.rmtree(tmp_dir)


@router.post("/ingest", status_code=HTTP_202_ACCEPTED)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    kb_id: str = Form(..., description="Unique identifier for the KB"),
    kb_name: str = Form(..., description="Human-readable name"),
    kb_description: str = Form(..., description="Summary of topics/domain"),
    files: List[UploadFile] = File(..., description="Files to ingest"),
    service: IngestionService = Depends(get_ingestion_service),
):
    """
    Accepts documents and metadata to trigger background ingestion.

    1. Registers/Updates Knowledge Base metadata in the registry.
    2. Saves uploaded files to a temporary location.
    3. Dispatches a background task for processing.
    """
    # 1. Register KB metadata (upsert to kb_registry)
    await service.register_kb(kb_id, kb_name, kb_description)

    # 2. Security validation & Save files to temporary directory
    tmp_dir = tempfile.mkdtemp()
    try:
        from src.config.settings import settings
        from fastapi import HTTPException
        from starlette.status import (
            HTTP_413_CONTENT_TOO_LARGE,
            HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

        for file in files:
            if not file.filename:
                continue

            # Ingestion Security: Extension Check
            extension = Path(file.filename).suffix.lstrip(".").lower()
            if extension not in settings.ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"File extension '{extension}' is not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
                )

            # Ingestion Security: File Size Check
            # Note: file.size is available since FastAPI 0.99.0
            max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
            if file.size and file.size > max_size_bytes:
                raise HTTPException(
                    status_code=HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File '{file.filename}' exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
                )

            # ponytail: prevent path traversal by taking only the filename
            safe_filename = Path(file.filename).name
            file_path = Path(tmp_dir) / safe_filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        shutil.rmtree(tmp_dir)
        raise e

    # 3. Trigger background processing
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_ingestion, service, tmp_dir, kb_id)

    return {"status": "accepted", "job_id": job_id, "kb_id": kb_id}
