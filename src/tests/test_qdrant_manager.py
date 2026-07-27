import pytest
import httpx
from unittest.mock import patch
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
from src.config.settings import settings
from src.utils.qdrant_manager import QdrantManager


@pytest.fixture
def mock_qdrant_client():
    with patch("src.utils.qdrant_manager.QdrantClient") as mock:
        yield mock


@pytest.mark.asyncio
async def test_qdrant_manager_initialization(mock_qdrant_client):
    """Test that QdrantManager initializes correctly with settings."""
    manager = QdrantManager(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )

    assert manager.client is not None
    mock_qdrant_client.assert_called_once_with(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )


@pytest.mark.asyncio
async def test_ensure_collection_creates_if_not_exists(mock_qdrant_client):
    """Test that ensure_collection creates a collection if it missing."""
    mock_instance = mock_qdrant_client.return_value
    mock_instance.collection_exists.return_value = False

    manager = QdrantManager(host="localhost", port=6333)
    await manager.ensure_collection("test_col", vector_size=1536)

    mock_instance.collection_exists.assert_called_once_with("test_col")
    mock_instance.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_skips_if_exists(mock_qdrant_client):
    """Test that ensure_collection skips creation if collection exists."""
    mock_instance = mock_qdrant_client.return_value
    mock_instance.collection_exists.return_value = True

    manager = QdrantManager(host="localhost", port=6333)
    await manager.ensure_collection("test_col", vector_size=1536)

    mock_instance.collection_exists.assert_called_once_with("test_col")
    mock_instance.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_points(mock_qdrant_client):
    """Test that upsert_points calls the client with expected arguments."""
    mock_instance = mock_qdrant_client.return_value
    manager = QdrantManager(host="localhost", port=6333)

    points = [PointStruct(id=1, vector=[0.1] * 1536, payload={"kb_id": "kb1"})]

    await manager.upsert_points("test_col", points)

    mock_instance.upsert.assert_called_once_with(
        collection_name="test_col", points=points
    )


@pytest.mark.asyncio
async def test_search_with_kb_id_filter(mock_qdrant_client):
    """Test that search applies the correct kb_id filter."""
    mock_instance = mock_qdrant_client.return_value
    from unittest.mock import MagicMock

    mock_instance.query_points.return_value = MagicMock(points=[])
    manager = QdrantManager(host="localhost", port=6333)

    query_vector = [0.1] * 1536
    kb_id = "target_kb"

    await manager.search("test_col", query_vector, kb_id=kb_id)

    expected_filter = Filter(
        must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
    )

    mock_instance.query_points.assert_called_once_with(
        collection_name="test_col",
        query=query_vector,
        query_filter=expected_filter,
        limit=5,
    )


@pytest.mark.asyncio
async def test_ensure_collection_handles_error(mock_qdrant_client):
    """Test that ensure_collection propagates connection errors."""
    mock_instance = mock_qdrant_client.return_value
    mock_instance.collection_exists.side_effect = UnexpectedResponse(
        500, "Internal Server Error", b"", httpx.Headers()
    )

    manager = QdrantManager(host="localhost", port=6333)

    with pytest.raises(UnexpectedResponse):
        await manager.ensure_collection("test_col", vector_size=1536)


@pytest.mark.asyncio
async def test_upsert_points_retries_on_failure(mock_qdrant_client):
    """Test that upsert_points retries on transient failures."""
    mock_instance = mock_qdrant_client.return_value
    # Fail twice, then succeed
    mock_instance.upsert.side_effect = [
        UnexpectedResponse(503, "Service Unavailable", b"", httpx.Headers()),
        UnexpectedResponse(503, "Service Unavailable", b"", httpx.Headers()),
        None,
    ]

    manager = QdrantManager(host="localhost", port=6333)
    points = [PointStruct(id=1, vector=[0.1] * 1536, payload={})]

    await manager.upsert_points("test_col", points)

    assert mock_instance.upsert.call_count == 3
