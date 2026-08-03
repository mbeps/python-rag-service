import pytest
from unittest.mock import MagicMock, AsyncMock
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from src.utils.qdrant_manager import QdrantManager


@pytest.fixture
def mock_async_qdrant_client():
    mock = MagicMock()
    mock.collection_exists = AsyncMock()
    mock.get_collection = AsyncMock()
    mock.create_collection = AsyncMock()
    mock.upsert = AsyncMock()
    mock.query_points = AsyncMock()
    mock.scroll = AsyncMock()
    mock.delete = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_qdrant_manager_initialization(mock_async_qdrant_client):
    """Test that QdrantManager initializes correctly."""
    manager = QdrantManager(client=mock_async_qdrant_client)
    assert manager.client == mock_async_qdrant_client


@pytest.mark.asyncio
async def test_ensure_collection_creates_if_not_exists(mock_async_qdrant_client):
    """Test that ensure_collection creates a collection if missing."""
    mock_async_qdrant_client.collection_exists.return_value = False
    manager = QdrantManager(client=mock_async_qdrant_client)

    await manager.ensure_collection("test_col", vector_size=1536)

    mock_async_qdrant_client.collection_exists.assert_called_once_with("test_col")
    mock_async_qdrant_client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_skips_if_exists(mock_async_qdrant_client):
    """Test that ensure_collection skips creation if collection exists with matching size."""
    mock_async_qdrant_client.collection_exists.return_value = True
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 1536
    mock_async_qdrant_client.get_collection.return_value = mock_collection_info

    manager = QdrantManager(client=mock_async_qdrant_client)
    await manager.ensure_collection("test_col", vector_size=1536)

    mock_async_qdrant_client.collection_exists.assert_called_once_with("test_col")
    mock_async_qdrant_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_points(mock_async_qdrant_client):
    """Test that upsert_points calls the client with expected arguments."""
    manager = QdrantManager(client=mock_async_qdrant_client)
    points = [PointStruct(id=1, vector=[0.1] * 1536, payload={"kb_id": "kb1"})]

    await manager.upsert_points("test_col", points)

    mock_async_qdrant_client.upsert.assert_called_once_with(
        collection_name="test_col", points=points
    )


@pytest.mark.asyncio
async def test_search_with_kb_id_filter(mock_async_qdrant_client):
    """Test that search applies the correct kb_id filter."""
    mock_async_qdrant_client.query_points.return_value = MagicMock(points=[])
    manager = QdrantManager(client=mock_async_qdrant_client)

    query_vector = [0.1] * 1536
    kb_id = "target_kb"

    await manager.search("test_col", query_vector, kb_id=kb_id)

    expected_filter = Filter(
        must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
    )

    mock_async_qdrant_client.query_points.assert_called_once_with(
        collection_name="test_col",
        query=query_vector,
        query_filter=expected_filter,
        limit=5,
    )
