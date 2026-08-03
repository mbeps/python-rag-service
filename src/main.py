"""Agentic RAG Service entry point."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, NotFoundError
from qdrant_client import AsyncQdrantClient

from src.api.v1.api import api_router
from src.config.settings import settings
from src.utils.minio_manager import MinIOManager
from src.utils.openai_client import get_openai_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes and verifies shared async clients.
    """
    logger.info("Starting up Agentic RAG Service...")

    # Initialize Clients
    app.state.qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )

    app.state.openai_client = get_openai_client()

    app.state.minio_manager = MinIOManager(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    # Verify Qdrant connectivity
    try:
        await app.state.qdrant_client.get_collections()
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")

    # Verify MinIO connectivity
    try:
        await app.state.minio_manager.list_buckets()
        logger.info("Successfully connected to MinIO.")
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")

    # Verify OpenAI-compatible provider connectivity AND authentication
    try:
        client = app.state.openai_client
        try:
            await client.get("/key", cast_to=httpx.Response)
        except (NotFoundError, Exception):
            # Fallback to models list if /key is not available
            await client.models.list()

        logger.info(
            "Successfully connected to OpenAI-compatible provider (authenticated)."
        )

        # Probe embedding model
        await client.embeddings.create(
            input="probe",
            model=settings.EMBEDDING_MODEL,
            encoding_format="float",
        )
        logger.info(f"Embedding model '{settings.EMBEDDING_MODEL}' is reachable.")
    except AuthenticationError:
        logger.error("OpenAI-compatible provider rejected the API key (401).")
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI-compatible provider: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Agentic RAG Service...")
    await app.state.qdrant_client.close()
    await app.state.openai_client.close()


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
