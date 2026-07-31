import pytest
from unittest.mock import patch
from io import BytesIO
from src.config.settings import settings
from src.utils.minio_manager import MinIOManager


@pytest.fixture
def mock_minio_client():
    with patch("src.utils.minio_manager.Minio") as mock:
        yield mock


@pytest.mark.asyncio
async def test_minio_manager_initialization(mock_minio_client):
    """Test that MinIOManager initializes correctly with settings."""
    manager = MinIOManager(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    assert manager.client is not None
    mock_minio_client.assert_called_once_with(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


@pytest.mark.asyncio
async def test_ensure_bucket_creates_if_missing(mock_minio_client):
    """Test that ensure_bucket creates a bucket if it doesn't exist."""
    mock_instance = mock_minio_client.return_value
    mock_instance.bucket_exists.return_value = False

    manager = MinIOManager("localhost:9000", "access", "secret")
    await manager.ensure_bucket("test-bucket")

    mock_instance.bucket_exists.assert_called_once_with("test-bucket")
    mock_instance.make_bucket.assert_called_once_with("test-bucket")


@pytest.mark.asyncio
async def test_ensure_bucket_skips_if_exists(mock_minio_client):
    """Test that ensure_bucket skips creation if bucket already exists."""
    mock_instance = mock_minio_client.return_value
    mock_instance.bucket_exists.return_value = True

    manager = MinIOManager("localhost:9000", "access", "secret")
    await manager.ensure_bucket("test-bucket")

    mock_instance.bucket_exists.assert_called_once_with("test-bucket")
    mock_instance.make_bucket.assert_not_called()


@pytest.mark.asyncio
async def test_upload_file(mock_minio_client):
    """Test that upload_file correctly calls put_object."""
    mock_instance = mock_minio_client.return_value
    manager = MinIOManager("localhost:9000", "access", "secret")

    data = b"test content"
    bucket_name = "test-bucket"
    object_name = "test.txt"
    content_type = "text/plain"

    await manager.upload_file(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type=content_type,
    )

    mock_instance.put_object.assert_called_once()
    args, kwargs = mock_instance.put_object.call_args
    assert args[0] == bucket_name
    assert args[1] == object_name
    assert args[3] == len(data)
    assert kwargs["content_type"] == content_type


@pytest.mark.asyncio
async def test_get_presigned_url(mock_minio_client):
    """Test that get_presigned_url returns a URL string."""
    mock_instance = mock_minio_client.return_value
    mock_instance.get_presigned_url.return_value = (
        "http://localhost:9000/test-bucket/test.txt?token"
    )

    manager = MinIOManager("localhost:9000", "access", "secret")
    url = await manager.get_presigned_url("test-bucket", "test.txt")

    assert "test-bucket/test.txt" in url
    mock_instance.get_presigned_url.assert_called_once()


@pytest.mark.asyncio
async def test_list_buckets(mock_minio_client):
    """Test that list_buckets returns a list of buckets."""
    mock_instance = mock_minio_client.return_value
    mock_instance.list_buckets.return_value = []

    manager = MinIOManager("localhost:9000", "access", "secret")
    buckets = await manager.list_buckets()

    assert isinstance(buckets, list)
    mock_instance.list_buckets.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_bucket_handles_error(mock_minio_client):
    """Test that ensure_bucket raises exception on connection failure."""
    mock_instance = mock_minio_client.return_value
    mock_instance.bucket_exists.side_effect = Exception("Connection failed")

    manager = MinIOManager("localhost:9000", "access", "secret")

    with pytest.raises(Exception, match="Connection failed"):
        await manager.ensure_bucket("test-bucket")


@pytest.mark.asyncio
async def test_upload_file_invalid_size(mock_minio_client):
    """Test that upload_file raises ValueError for zero or negative size."""
    manager = MinIOManager("localhost:9000", "access", "secret")

    with pytest.raises(ValueError, match="File size must be greater than zero"):
        await manager.upload_file(
            bucket_name="test", object_name="test", data=BytesIO(b""), length=0
        )
