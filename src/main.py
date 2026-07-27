"""Agentic RAG Service entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.api import api_router
from src.config.settings import settings
from src.utils.minio_manager import MinIOManager
from src.utils.qdrant_manager import QdrantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Ensures that Qdrant and MinIO are reachable on startup.
    """
    logger.info("Starting up Agentic RAG Service...")

    # Verify Qdrant connectivity
    try:
        qdrant = QdrantManager(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        qdrant.client.get_collections()
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        # In a real production app, we might want to fail hard here
        # raise e

    # Verify MinIO connectivity
    try:
        minio = MinIOManager(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        minio.client.list_buckets()
        logger.info("Successfully connected to MinIO.")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        # raise e

    yield
    logger.info("Shutting down Agentic RAG Service...")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
