from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ProcessedChunk(BaseModel):
    """
    Represents a chunk of text with associated metadata and optional visual assets.

    Attributes:
        text (str): The text content of the chunk.
        image_url (Optional[str]): Presigned URL to the associated image or table in MinIO.
        metadata (Dict[str, Any]): Dictionary of metadata extracted from the document.
        kb_id (str): The ID of the knowledge base this chunk belongs to.
    """

    text: str = Field(..., description="The text content of the chunk.")
    image_url: Optional[str] = Field(
        None, description="Presigned URL to the associated image or table."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata extracted from the document."
    )
    kb_id: str = Field(..., description="The ID of the knowledge base.")
    document_id: str = Field(..., description="Stable identifier for the document.")
    chunk_id: str = Field(..., description="Deterministic unique identifier for the chunk.")
