import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from langchain_core.runnables import RunnableConfig
from qdrant_client.http.models import ScoredPoint
from openai import AsyncOpenAI

from src.config.settings import get_settings, Settings
from src.utils.qdrant_manager import QdrantManager
from src.agent.graph import app as agent_app
from src.schemas.query_request import QueryRequest
from src.schemas.query_response import QueryResponse
from src.schemas.agent_state import AgentState
from src.api.v1.dependencies import get_qdrant_manager, get_openai_client

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
) -> QueryResponse:
    """
    Standard user query interface for the RAG service.

    Performs dynamic KB selection if kb_id is missing, then invokes
    the LangGraph agent to generate a grounded response.
    """
    kb_id = request.kb_id

    # 1. Dynamic KB Selection if kb_id is missing
    if not kb_id:
        # Embed the query and search kb_registry for a matching knowledge base
        response = await openai_client.embeddings.create(
            input=request.query,
            model=settings.EMBEDDING_MODEL,
            encoding_format="float",
        )
        query_vector = response.data[0].embedding
        matches: List[ScoredPoint] = await qdrant.search(
            collection_name=settings.KB_REGISTRY_COLLECTION,
            query_vector=query_vector,
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
    initial_state = AgentState(
        query=request.query,
        kb_id=kb_id,
        use_multimodal=request.use_multimodal,
        documents=[],
        citations=[],
        visual_references=[],
        answer=None,
        rewritten_query=None,
        grader_feedback=None,
    )

    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        final_state = await agent_app.ainvoke(initial_state, config=config)

        if isinstance(final_state, dict):
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
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
