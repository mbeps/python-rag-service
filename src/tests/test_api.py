import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO


@pytest.fixture
def mock_indexer():
    """Mock the IngestionService methods."""
    with (
        patch("src.api.v1.dependencies.IngestionService") as mock_ingest_class,
    ):
        mock_instance = mock_ingest_class.return_value
        mock_instance.ingest = AsyncMock()
        mock_instance.register_kb = AsyncMock()
        yield mock_instance.ingest, mock_instance.register_kb


@pytest.fixture
def mock_qdrant_manager():
    """Mock QdrantManager for KB registry operations."""
    with (
        patch("src.api.v1.dependencies.QdrantManager") as mock_manager_class,
    ):
        mock_instance = mock_manager_class.return_value
        mock_instance.ensure_collection = AsyncMock()
        mock_instance.upsert_points = AsyncMock()
        mock_instance.search = AsyncMock()
        yield mock_instance


@pytest.fixture
def mock_agent_app():
    """Mock the LangGraph agent app."""
    with patch("src.api.v1.endpoints.query.agent_app") as mock:
        mock.ainvoke = AsyncMock()
        yield mock


def test_health_check(client_no_lifespan):
    """Test the health check endpoint."""
    response = client_no_lifespan.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_documents_success(
    client_no_lifespan, mock_indexer, mock_qdrant_manager
):
    """Test successful document ingestion trigger."""
    mock_ingest, mock_reg = mock_indexer
    files = [
        ("files", ("test1.pdf", BytesIO(b"dummy pdf content"), "application/pdf")),
        ("files", ("test2.txt", BytesIO(b"dummy text content"), "text/plain")),
    ]
    data = {
        "kb_id": "test_kb",
        "kb_name": "Test Knowledge Base",
        "kb_description": "A KB for specialized testing",
    }

    response = client_no_lifespan.post("/api/v1/ingest", data=data, files=files)

    assert response.status_code == 202
    assert "job_id" in response.json()
    assert response.json()["status"] == "accepted"

    # Verify background task was called (conceptually)
    # mock_ingest is called in background, but the endpoint should have planned it
    # Since we are mocking the class method, and BackgroundTasks is scheduled by FastAPI,
    # we can't easily check if it was CALLED yet unless we wait or mock BackgroundTasks.
    # But registering metadata is synchronous (awaited) in the endpoint before return.
    mock_reg.assert_called_once()


def test_ingest_documents_missing_files(client_no_lifespan):
    """Test ingestion fails when no files are provided."""
    data = {"kb_id": "test_kb", "kb_name": "Test KB", "kb_description": "Desc"}
    response = client_no_lifespan.post("/api/v1/ingest", data=data)
    # FastAPI returns 422 for missing required fields (if files is required)
    assert response.status_code == 422


def test_query_explicit_kb_success(client_no_lifespan, mock_agent_app):
    """Test query with an explicit kb_id."""
    mock_agent_app.ainvoke.return_value = {
        "answer": "The answer is 42.",
        "citations": [{"source": "docs/test.pdf", "page": 1}],
        "visual_assets": [],
        "kb_id": "test_kb",
    }

    query_data = {
        "query": "What is the answer?",
        "kb_id": "test_kb",
        "use_multimodal": True,
    }

    response = client_no_lifespan.post("/api/v1/query", json=query_data)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer is 42."
    assert data["kb_id"] == "test_kb"
    mock_agent_app.ainvoke.assert_called_once()


def test_query_dynamic_kb_selection(
    client_no_lifespan, mock_agent_app, mock_qdrant_manager
):
    """Test that query performs dynamic KB selection when kb_id is missing."""
    # Mock Qdrant search to return a matching KB
    mock_match = MagicMock()
    mock_match.payload = {"kb_id": "dynamic_kb"}
    mock_match.score = 0.85
    mock_qdrant_manager.search.return_value = [mock_match]

    mock_agent_app.ainvoke.return_value = {
        "answer": "Dynamic answer.",
        "citations": [],
        "visual_assets": [],
        "kb_id": "dynamic_kb",
    }

    query_data = {
        "query": "Something that needs dynamic selection.",
    }

    response = client_no_lifespan.post("/api/v1/query", json=query_data)

    assert response.status_code == 200
    data = response.json()
    assert data["kb_id"] == "dynamic_kb"
    # Verify Qdrant search was called because kb_id was missing
    mock_qdrant_manager.search.assert_called_once()
    mock_agent_app.ainvoke.assert_called_once()


def test_query_dynamic_kb_no_match(client_no_lifespan, mock_qdrant_manager):
    """Test dynamic selection when no KB matches the threshold."""
    # Mock Qdrant search to return low score matches
    mock_match = MagicMock()
    mock_match.payload = {"kb_id": "irrelevant_kb"}
    mock_match.score = 0.3
    mock_qdrant_manager.search.return_value = [mock_match]

    query_data = {
        "query": "Queries with no relevant KB.",
    }

    response = client_no_lifespan.post("/api/v1/query", json=query_data)

    # Should probably return 404 or a specific error message if no KB matches
    assert response.status_code == 404
    assert "No relevant Knowledge Base found" in response.json()["detail"]


def test_query_invalid_schema(client_no_lifespan):
    """Test query fails with invalid request body."""
    response = client_no_lifespan.post("/api/v1/query", json={})  # Missing "query"
    assert response.status_code == 422
