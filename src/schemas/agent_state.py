from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Typed state for LangGraph state management.

    Attributes:
        query (str): Original user query.
        documents (List[Dict[str, Union[str, int, float, bool]]]): Retrieved document metadata and content.
        answer (Optional[str]): Final generated answer.
        kb_id (str): Target or selected Knowledge Base identifier.
        rewritten_query (Optional[str]): Optimized query for retrieval.
        grader_feedback (Optional[str]): Feedback from the document/answer grader.
        visual_references (List[Dict[str, Union[str, int, float, bool]]]): Data for visual assets.
    """

    query: str = Field(..., description="Original user query")
    documents: List[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=list, description="Retrieved documents"
    )
    answer: Optional[str] = Field(None, description="Final generated answer")
    kb_id: str = Field(..., description="Selected Knowledge Base identifier")
    rewritten_query: Optional[str] = Field(None, description="Rewritten query")
    grader_feedback: Optional[str] = Field(None, description="Feedback from grader")
    visual_references: List[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=list, description="References to visual assets"
    )
