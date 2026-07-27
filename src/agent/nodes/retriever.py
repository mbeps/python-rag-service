from typing import cast

from src.config.settings import get_settings
from src.utils.qdrant_manager import QdrantManager
from src.schemas.agent_state import AgentState


async def retriever_node(state: AgentState) -> AgentState:
    """Retrieves relevant documents from the knowledge base.

    Args:
        state (AgentState): The current state of the agent, including
            'query', 'kb_id', and optionally 'rewritten_query'.

    Returns:
        AgentState: The updated state with 'documents' populated
            from the search results.
    """
    settings = get_settings()

    # Initialize QdrantManager
    qdrant_manager = QdrantManager(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )

    # Select query: rewritten_query takes precedence
    # We support both dict and AgentState object for robustness in LangGraph
    if isinstance(state, dict):
        query = state.get("rewritten_query") or state.get("query")
        kb_id = state.get("kb_id")
    else:
        query = state.rewritten_query or state.query
        kb_id = state.kb_id

    # Perform search
    # ponytail: search currently assumes 'knowledge_base' collection.
    results = await qdrant_manager.search(
        collection_name="knowledge_base", query=cast(str, query), kb_id=kb_id, limit=5
    )

    # Convert ScoredPoint results to metadata dicts for the state
    documents = [point.payload for point in results if point.payload]

    # Update and return state
    if isinstance(state, dict):
        state["documents"] = documents
    else:
        state.documents = documents

    return state
