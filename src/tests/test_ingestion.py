from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from src.ingestion.partitioner import DocumentPartitioner
from src.ingestion.indexer import IngestionService
from src.schemas.processed_chunk import ProcessedChunk


def test_partition_and_chunk():
    """
    Test the partition_and_chunk method of DocumentPartitioner by mocking unstructured functions.
    """
    # Arrange
    partitioner = DocumentPartitioner()
    mock_file_path = Path("/tmp/sample.pdf")

    # Create mock elements
    mock_element = MagicMock()
    mock_element.text = "Hello world"
    mock_elements = [mock_element]

    # Create mock chunks
    mock_chunk = MagicMock()
    mock_chunk.text = "Hello world"
    mock_chunks = [mock_chunk]

    with patch(
        "src.ingestion.partitioner.partition_pdf", return_value=mock_elements
    ) as mock_partition:
        with patch(
            "src.ingestion.partitioner.chunk_by_title", return_value=mock_chunks
        ) as mock_chunk_title:
            # Act
            result = partitioner.partition_and_chunk(mock_file_path)

            # Assert
            mock_partition.assert_called_once_with(
                filename=str(mock_file_path),
                pdf_infer_table_structure=True,
                strategy="fast",
            )
            mock_chunk_title.assert_called_once()
            assert len(result) == 1
            assert result[0].text == "Hello world"


@pytest.mark.asyncio
async def test_ingestion_service_full_flow():
    """
    Test the full ingestion pipeline in IngestionService.
    """
    # Arrange
    mock_settings = MagicMock()
    mock_settings.OPENAI_API_KEY = "test_key"
    mock_settings.OPENAI_BASE_URL = "test_url"
    mock_settings.EMBEDDING_MODEL = "test_model"
    mock_settings.EMBEDDING_DIMENSIONS = 1536

    mock_qdrant = AsyncMock()
    mock_minio = MagicMock()
    kb_id = "test_kb"
    file_path = Path("/tmp/test.pdf")

    # Mock chunks
    mock_chunk = MagicMock()
    mock_chunk.text = "chunk text"

    # Mock processed chunks
    processed_chunk = ProcessedChunk(
        text="chunk text",
        image_url="http://test.com/img.png",
        metadata={"page": 1},
        kb_id=kb_id,
    )

    # Mock response from OpenAI
    mock_embedding_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536
    mock_embedding_response.data = [mock_embedding]

    with (
        patch("src.ingestion.indexer.DocumentPartitioner") as MockPartitioner,
        patch("src.ingestion.indexer.AssetProcessor") as MockProcessor,
        patch("src.utils.openai_client.AsyncOpenAI") as MockOpenAI,
    ):
        # Setup mocks
        MockPartitioner.return_value.partition_and_chunk.return_value = [mock_chunk]
        MockProcessor.return_value.process_assets = AsyncMock(
            return_value=[processed_chunk]
        )

        mock_openai_instance = MockOpenAI.return_value
        mock_openai_instance.embeddings.create = AsyncMock(
            return_value=mock_embedding_response
        )

        # Initialize service inside the patch context
        service = IngestionService(mock_settings, mock_qdrant, mock_minio)

        # Act
        await service.ingest(file_path, kb_id)

        # Assert
        MockPartitioner.return_value.partition_and_chunk.assert_called_once_with(
            file_path
        )
        MockProcessor.return_value.process_assets.assert_called_once_with(
            [mock_chunk], kb_id
        )
        mock_qdrant.ensure_collection.assert_called_once()
        mock_qdrant.upsert_points.assert_called_once()

        # Verify OpenAI call
        mock_openai_instance.embeddings.create.assert_called_once_with(
            input="chunk text", model="test_model", encoding_format="float"
        )

        _, kwargs = mock_qdrant.upsert_points.call_args
        points = kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["text"] == "chunk text"
        assert points[0].payload["kb_id"] == kb_id
        assert points[0].payload["image_url"] == "http://test.com/img.png"
        assert points[0].payload["page"] == 1
