from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    """Parameters for document ingestion.

    Attributes:
        kb_id (str): Target Knowledge Base identifier.
        files (List[str]): List of file paths or identifiers to process.
        metadata (Optional[Dict[str, Union[str, int, float, bool]]]): Extra metadata for the session.
    """

    kb_id: str = Field(..., description="Target Knowledge Base identifier")
    files: List[str] = Field(..., description="List of file paths or identifiers")
    metadata: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        None, description="Extra metadata for the session"
    )
