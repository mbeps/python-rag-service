"""Pytest configuration and shared fixtures for the test suite.

This module provides centralized test fixtures for the RAG service, including:
- app_without_lifespan: FastAPI app with lifespan disabled to prevent real API calls
- mock_clients: Mocked async clients for Qdrant, OpenAI, and MinIO
- client_no_lifespan: TestClient using the test app with mocked clients
- client: Legacy fixture for backward compatibility (makes real API calls)
"""

import pytest
from typing import Generator
from unittest.mock import AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from src.main import app as main_app
from src.config.settings import settings


@pytest.fixture
def app_without_lifespan() -> FastAPI:
    """
    Create a FastAPI app instance with lifespan disabled.

    This prevents startup/shutdown hooks from making real API calls to Qdrant,
    OpenAI, and MinIO during test initialization. The app includes the same
    router and middleware as the main app.

    Returns:
        FastAPI: Test app with lifespan=None
    """
    test_app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        lifespan=None,  # Disable lifespan for tests
    )

    # Add CORS middleware if configured
    if settings.BACKEND_CORS_ORIGINS:
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Copy router from main app
    test_app.include_router(main_app.router)

    # Add health check endpoint
    @test_app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return test_app


@pytest.fixture
def mock_clients() -> dict:
    """
    Create mock instances for all async clients used in app.state.

    Returns:
        dict: Dictionary mapping client names to AsyncMock instances:
            - "qdrant_client": Mock for Qdrant vector database
            - "openai_client": Mock for OpenAI-compatible API
            - "minio_manager": Mock for MinIO object storage
    """
    return {
        "qdrant_client": AsyncMock(),
        "openai_client": AsyncMock(),
        "minio_manager": AsyncMock(),
    }


@pytest.fixture
def client_no_lifespan(
    app_without_lifespan, mock_clients
) -> Generator[TestClient, None, None]:
    """
    Test client with lifespan disabled and mocked external clients.

    This fixture creates a TestClient that doesn't execute the app's lifespan
    startup events. It injects mock clients into app.state, allowing tests to
    run without making real API calls to external services.

    Use this fixture in most tests unless you specifically need to test
    lifespan behavior or make real API calls.

    Args:
        app_without_lifespan: FastAPI app with lifespan disabled
        mock_clients: Dictionary of mocked async clients

    Yields:
        TestClient: FastAPI TestClient with mocked clients
    """
    # Inject mock clients into app.state
    for key, mock_client in mock_clients.items():
        setattr(app_without_lifespan.state, key, mock_client)

    with TestClient(app_without_lifespan) as c:
        yield c


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    Default test client with full lifespan (legacy).

    WARNING: This fixture makes REAL API calls to external services during
    startup (Qdrant, OpenAI, MinIO). Tests using this fixture will:
    - Be slower due to network latency
    - Consume API quota
    - Fail if services are unavailable or credentials are invalid

    Use `client_no_lifespan` instead for most tests. Only use this fixture
    if you specifically need to test lifespan behavior or integration with
    real services.

    Yields:
        TestClient: FastAPI TestClient using the main app with full lifespan
    """
    with TestClient(main_app) as c:
        yield c
