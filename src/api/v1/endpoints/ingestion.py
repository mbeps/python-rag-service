import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, Depends
from starlette.status import HTTP_202_ACCEPTED

from src.config.settings import Settings, get_settings
from src.ingestion.indexer import IngestionService
from src.utils.minio_manager import MinIOManager
from src.utils.qdrant_manager import QdrantManager

router = APIRouter()


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
) -> IngestionService:
    """Dependency injection for IngestionService."""
    qdrant_manager = QdrantManager(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )
    minio_manager = MinIOManager(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )
    return IngestionService(settings, qdrant_manager, minio_manager)


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

    # 2. Save files to temporary directory
    tmp_dir = tempfile.mkdtemp()
    for file in files:
        if not file.filename:
            continue
        file_path = Path(tmp_dir) / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # 3. Trigger background processing
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_ingestion, service, tmp_dir, kb_id)

    return {"status": "accepted", "job_id": job_id, "kb_id": kb_id}
