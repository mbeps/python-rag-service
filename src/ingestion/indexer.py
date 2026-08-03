import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from qdrant_client.http.models import PointStruct

from src.config.settings import Settings
from src.ingestion.asset_processor import AssetProcessor
from src.ingestion.partitioner import DocumentPartitioner
from src.utils.minio_manager import MinIOManager
from src.utils.openai_client import get_openai_client
from src.utils.qdrant_manager import QdrantManager


class IngestionService:
    """
    Orchestrates the ingestion pipeline from partitioning to vector storage.
    Handles partitioning, asset extraction/storage, embedding generation, and indexing.
    """

    def __init__(
        self,
        settings: Settings,
        metadata_manager: QdrantManager,
        asset_manager: MinIOManager,
    ) -> None:
        """
        Initializes the IngestionService with dependencies.

        Args:
            settings (Settings): Application configuration.
            metadata_manager (QdrantManager): Vector database manager.
            asset_manager (MinIOManager): Object storage manager.
        """
        self.settings = settings
        self.metadata_manager = metadata_manager
        self.asset_manager = asset_manager

        self.openai_client = get_openai_client()
        # ponytail: use the collection name from settings.
        self.collection_name = self.settings.QDRANT_COLLECTION

    async def register_kb(self, kb_id: str, name: str, description: str) -> None:
        """
        Registers or updates a Knowledge Base in the global registry.
        Generates an embedding for the description to allow dynamic KB selection.

        Args:
            kb_id (str): Unique identifier for the KB.
            name (str): Human-readable name.
            description (str): Summary of topics/domain.
        """
        await self.metadata_manager.ensure_collection(
            collection_name=self.settings.KB_REGISTRY_COLLECTION,
            vector_size=self.settings.EMBEDDING_DIMENSIONS,
        )

        response = await self.openai_client.embeddings.create(
            input=description,
            model=self.settings.EMBEDDING_MODEL,
            encoding_format="float",
        )
        vector = response.data[0].embedding

        payload = {
            "kb_id": kb_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }

        # Deterministic UUID for the KB in the registry
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, kb_id))

        await self.metadata_manager.upsert_points(
            collection_name=self.settings.KB_REGISTRY_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def ingest(self, file_path: Path, kb_id: str) -> None:
        """
        Executes the full ingestion pipeline for a single file.

        Steps:
        1. Partition and chunk the document using layout awareness.
        2. Process visual assets (images, tables) and upload to MinIO.
        3. Generate vector embeddings for the text content.
        4. Index chunks and metadata in Qdrant.

        Args:
            file_path (Path): Path to the source document.
            kb_id (str): Unique identifier for the knowledge base.

        Returns:
            None
        """
        # 0. Generate Document ID and handle idempotency
        file_content = file_path.read_bytes()
        document_id = hashlib.sha256(file_content).hexdigest()

        # Idempotency: Delete existing chunks of this document in this KB
        await self.metadata_manager.delete_points_by_filter(
            collection_name=self.collection_name,
            kb_id=kb_id,
            document_id=document_id,
        )

        # 1. Partition & Chunk
        partitioner = DocumentPartitioner()
        chunks = partitioner.partition_and_chunk(file_path)

        # 2. Process Assets (Images/Tables)
        processor = AssetProcessor(self.asset_manager)
        processed_chunks = await processor.process_assets(chunks, kb_id)

        # 3. Generate Embeddings & Create Qdrant Points
        await self.metadata_manager.ensure_collection(
            collection_name=self.collection_name,
            vector_size=self.settings.EMBEDDING_DIMENSIONS,
        )

        points = []
        for i, chunk in enumerate(processed_chunks):
            # Generate deterministic chunk ID
            chunk_id = str(uuid.uuid5(uuid.UUID(hex=document_id[:32]), str(i)))

            # Generate embedding using OpenAI-compatible API
            response = await self.openai_client.embeddings.create(
                input=chunk.text,
                model=self.settings.EMBEDDING_MODEL,
                encoding_format="float",
            )
            vector = response.data[0].embedding

            # Prepare metadata payload
            payload = {
                "text": chunk.text,
                "kb_id": kb_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "image_url": chunk.image_url,
                **chunk.metadata,
            }

            points.append(PointStruct(id=chunk_id, vector=vector, payload=payload))

        # 4. Batch Upsert to Qdrant
        if points:
            await self.metadata_manager.upsert_points(
                collection_name=self.collection_name, points=points
            )
