from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User query parameters for the RAG service.

    Attributes:
        query (str): The user question.
        kb_id (Optional[str]): Manual Knowledge Base selection. If None, triggers dynamic routing.
        use_multimodal (bool): Whether to include visual assets in reasoning. Defaults to True.
    """

    query: str = Field(..., description="The user question")
    kb_id: Optional[str] = Field(
        None, description="Manual KB selection; triggers dynamic routing if None"
    )
    use_multimodal: bool = Field(
        True, description="Whether to include visual assets in reasoning"
    )
