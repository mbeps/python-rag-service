import io
import uuid
from typing import List, Any, Optional
from unstructured.documents.elements import Image, Table

from src.schemas.processed_chunk import ProcessedChunk
from src.utils.minio_manager import MinIOManager


class AssetProcessor:
    """
    Processes extracted images and tables, manages storage and visual metadata.
    Enriches chunks with visual asset information and descriptions.
    """

    def __init__(
        self, minio_manager: MinIOManager, vision_llm_settings: Optional[dict] = None
    ) -> None:
        """
        Initializes the AssetProcessor.

        Args:
            minio_manager (MinIOManager): The manager for MinIO operations.
            vision_llm_settings (Optional[dict]): Settings for the Vision LLM.
        """
        self.minio_manager = minio_manager
        self.vision_llm_settings = vision_llm_settings or {}
        self.visual_bucket = "visual-assets"

    async def process_assets(
        self, chunks: List[Any], kb_id: str
    ) -> List[ProcessedChunk]:
        """
        Iterates through chunks, detects visual elements, uploads them to MinIO,
        and generates captions/summaries.

        Args:
            chunks (List[Any]): List of chunks from the partitioner.
            kb_id (str): The ID of the knowledge base.

        Returns:
            List[ProcessedChunk]: List of processed chunks ready for indexing.
        """
        await self.minio_manager.ensure_bucket(self.visual_bucket)
        processed_chunks = []

        for chunk in chunks:
            text = getattr(chunk, "text", "")
            metadata = getattr(chunk, "metadata", {})
            # convert metadata to dict if it's an object (unstructured metadata objects)
            if hasattr(metadata, "to_dict"):
                metadata = metadata.to_dict()

            image_url = None

            # Check for visual elements in the chunk
            # 'unstructured' chunks often have 'orig_elements' in metadata
            orig_elements = metadata.get("orig_elements", [])

            # If the chunk itself is an Image or Table, or contains them
            visual_element = None
            if isinstance(chunk, (Image, Table)):
                visual_element = chunk
            elif orig_elements:
                for el in orig_elements:
                    if isinstance(el, (Image, Table)):
                        visual_element = el
                        break

            if visual_element:
                asset_id = str(uuid.uuid4())
                asset_type = "image" if isinstance(visual_element, Image) else "table"
                page_number = metadata.get("page_number", "unknown")

                # Mock caption/summary
                # ponytail: mock captioning for now. upgrade: integrate Vision LLM.
                caption = f"Description of {asset_type} at page {page_number}"
                text = f"{text}\n\n[Visual Asset Description: {caption}]"

                # Save to temporary file and upload
                # In unstructured, images/tables might have 'image_path' or bytes if extracted.
                # For now, we simulate extraction if bytes are available, otherwise we use a placeholder.
                # ponytail: using placeholder if pixels are not available.
                data = b"placeholder_image_data"
                if (
                    hasattr(visual_element, "image_bytes")
                    and visual_element.image_bytes
                ):
                    data = visual_element.image_bytes

                object_name = f"{kb_id}/{asset_id}.png"
                data_stream = io.BytesIO(data)

                await self.minio_manager.upload_file(
                    bucket_name=self.visual_bucket,
                    object_name=object_name,
                    data=data_stream,
                    length=len(data),
                    content_type="image/png",
                )

                image_url = await self.minio_manager.get_presigned_url(
                    bucket_name=self.visual_bucket, object_name=object_name
                )

            processed_chunks.append(
                ProcessedChunk(
                    text=text,
                    image_url=image_url,
                    metadata=metadata,
                    kb_id=kb_id,
                    document_id=metadata.get("document_id", "pending"),
                    chunk_id=metadata.get("chunk_id", "pending"),
                )
            )

        return processed_chunks
