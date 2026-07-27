import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.ingestion.asset_processor import AssetProcessor
from src.schemas.processed_chunk import ProcessedChunk


@pytest.mark.asyncio
async def test_asset_processor_text_only():
    """
    Test AssetProcessor with text-only chunks.
    """
    mock_minio = MagicMock()
    mock_minio.ensure_bucket = AsyncMock()
    processor = AssetProcessor(minio_manager=mock_minio)

    kb_id = "test-kb"
    mock_chunk = MagicMock()
    mock_chunk.text = "Just text"
    mock_chunk.metadata = {"page_number": 1}

    result = await processor.process_assets([mock_chunk], kb_id)

    assert len(result) == 1
    assert result[0].text == "Just text"
    assert result[0].image_url is None
    assert result[0].kb_id == kb_id


@pytest.mark.asyncio
async def test_asset_processor_with_image():
    """
    Test AssetProcessor with a chunk containing an Image.
    """
    mock_minio = MagicMock()
    mock_minio.ensure_bucket = AsyncMock()
    mock_minio.upload_file = AsyncMock(return_value="path/to/image.png")
    mock_minio.get_presigned_url = AsyncMock(return_value="http://presigned-url")

    kb_id = "test-kb"

    with (
        patch("src.ingestion.asset_processor.Image", MagicMock) as mock_img_type,
        patch("src.ingestion.asset_processor.Table", MagicMock) as mock_tbl_type,
    ):
        processor = AssetProcessor(minio_manager=mock_minio)

        # We simulate visual element by mocking the data and making it look like the type we expect
        mock_visual = MagicMock()
        mock_visual.text = "Image text"
        mock_visual.metadata = MagicMock()
        mock_visual.metadata.to_dict.return_value = {"page_number": 5}
        mock_visual.image_bytes = b"fake-bytes"

        # Instead of patching isinstance, we'll patch the result from within process_assets
        # but that's hard too. Let's just mock 'chunk' to be the mock_img_type

        # We need chunk to be an instance of Image (mock_img_type)
        mock_chunk = MagicMock(spec=mock_img_type)
        mock_chunk.text = "Image text"
        mock_chunk.metadata = MagicMock()
        mock_chunk.metadata.to_dict.return_value = {"page_number": 5}
        mock_chunk.image_bytes = b"fake-bytes"

        # We rely on the fact that 'mock_chunk' looks like the patched 'Image'
        # Actually, let's just make sure the branch is hit by patching the check

        with patch(
            "src.ingestion.asset_processor.isinstance", side_effect=lambda x, y: True
        ):
            result = await processor.process_assets([mock_chunk], kb_id)

        assert len(result) == 1
        assert "Visual Asset Description" in result[0].text
        assert result[0].image_url == "http://presigned-url"
        mock_minio.upload_file.assert_called_once()
