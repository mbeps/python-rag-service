from fastapi import Request
from src.utils.qdrant_manager import QdrantManager
from src.utils.minio_manager import MinIOManager
from openai import AsyncOpenAI
from src.config.settings import Settings, get_settings
from src.ingestion.indexer import IngestionService


async def get_qdrant_manager(request: Request) -> QdrantManager:
    """Dependency injection for QdrantManager using the shared client."""
    return QdrantManager(client=request.app.state.qdrant_client)


async def get_minio_manager(request: Request) -> MinIOManager:
    """Dependency injection for MinIOManager."""
    return request.app.state.minio_manager


async def get_openai_client(request: Request) -> AsyncOpenAI:
    """Dependency injection for OpenAI client."""
    return request.app.state.openai_client


async def get_ingestion_service(
    request: Request,
    settings: Settings = get_settings(),
) -> IngestionService:
    """Dependency injection for IngestionService."""
    qdrant_manager = await get_qdrant_manager(request)
    minio_manager = await get_minio_manager(request)
    return IngestionService(settings, qdrant_manager, minio_manager)
