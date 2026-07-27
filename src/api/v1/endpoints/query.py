import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from qdrant_client.http.models import ScoredPoint

from src.config.settings import get_settings, Settings
from src.utils.qdrant_manager import QdrantManager
from src.agent.graph import app as agent_app
from src.schemas.query_request import QueryRequest
from src.schemas.query_response import QueryResponse
from src.schemas.agent_state import AgentState

router = APIRouter()


async def get_qdrant_manager(
    settings: Settings = Depends(get_settings),
) -> QdrantManager:
    """Dependency to provide a QdrantManager instance."""
    return QdrantManager(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
) -> QueryResponse:
    """
    Standard user query interface for the RAG service.

    Performs dynamic KB selection if kb_id is missing, then invokes
    the LangGraph agent to generate a grounded response.

    Args:
        request: The query request parameters.
        settings: Application settings.
        qdrant: Qdrant manager for vector searches.

    Returns:
        The agent's generated answer and metadata.

    Raises:
        HTTPException: If no matching KB is found when dynamic selection is triggered.
    """
    kb_id = request.kb_id

    # 1. Dynamic KB Selection if kb_id is missing
    if not kb_id:
        # Search kb_registry for a matching knowledge base
        # ponytail: Search assumes the 'query' string can be used as a vector
        # by Qdrant's query_points (FastEmbed/Auto-inference).
        matches: List[ScoredPoint] = await qdrant.search(
            collection_name=settings.KB_REGISTRY_COLLECTION,
            query=request.query,
            limit=1,
        )

        if (
            not matches
            or not matches[0].payload
            or matches[0].score < settings.DYNAMIC_KB_THRESHOLD
        ):
            raise HTTPException(
                status_code=404,
                detail="No relevant Knowledge Base found for the given query.",
            )

        kb_id = str(matches[0].payload.get("kb_id"))

    # 2. Invoke LangGraph Agent
    # Initial state for the agent
    initial_state = AgentState(
        query=request.query,
        kb_id=kb_id,
        use_multimodal=request.use_multimodal,
        documents=[],
        citations=[],
        visual_references=[],
    )

    # Invoke the compiled graph with a thread_id for state persistence
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        # ainvoke returns the final state
        final_state = await agent_app.ainvoke(initial_state, config=config)

        # ponytail: LangGraph ainvoke might return a dict or the state object
        # depending on version and configuration. We handle both.
        if isinstance(final_state, dict):
            # Map visual_references (dict with image_url) to visual_assets (list of URLs)
            visual_refs = final_state.get("visual_references", [])
            visual_assets = [
                ref["image_url"]
                for ref in visual_refs
                if isinstance(ref, dict) and "image_url" in ref
            ]
            return QueryResponse(
                answer=final_state.get("answer", ""),
                citations=final_state.get("citations", []),
                visual_assets=visual_assets,
                kb_id=kb_id,
            )
        else:
            # Map visual_references (dict with image_url) to visual_assets (list of URLs)
            visual_assets = [
                ref["image_url"]
                for ref in final_state.visual_references
                if isinstance(ref, dict) and "image_url" in ref
            ]
            return QueryResponse(
                answer=final_state.answer,
                citations=final_state.citations,
                visual_assets=visual_assets,
                kb_id=kb_id,
            )

    except Exception as e:
        # ponytail: simple error propagation.
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
