"""Agentic RAG Service entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, NotFoundError

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
        await asyncio.to_thread(qdrant.client.get_collections)
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
        await minio.list_buckets()
        logger.info("Successfully connected to MinIO.")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        # raise e

    # Verify OpenAI-compatible provider connectivity AND authentication
    try:
        from src.utils.openai_client import get_openai_client

        client = get_openai_client()
        try:
            # OpenRouter exposes an authenticated /key endpoint; generic
            # providers may not, so fall back to the models list (which is
            # authenticated on OpenAI itself).
            await client.get("/key", cast_to=httpx.Response)
        except NotFoundError:
            await client.models.list()
        logger.info(
            "Successfully connected to OpenAI-compatible provider (authenticated)."
        )

        # Probe the embedding model early to catch "model not found" at startup
        await client.embeddings.create(
            input="probe",
            model=settings.EMBEDDING_MODEL,
            encoding_format="float",
        )
        logger.info(f"Embedding model '{settings.EMBEDDING_MODEL}' is reachable.")
    except AuthenticationError:
        logger.error(
            "OpenAI-compatible provider rejected the API key (401). "
            "Set a valid OPENAI_API_KEY in your .env file (see .env.example)."
        )
    except Exception as e:
        logger.error(
            f"Failed to connect to OpenAI-compatible provider: {e}. "
            "Check OPENAI_API_KEY and EMBEDDING_MODEL in your .env file."
        )

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
