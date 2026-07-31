from typing import cast

from src.config.settings import get_settings
from src.schemas.agent_state import AgentState
from src.utils.openai_client import get_openai_client
from src.utils.qdrant_manager import QdrantManager


def _sanitise_document(
    payload: dict[str, object],
) -> dict[str, str | int | float | bool]:
    """Map a Qdrant payload to the fields the agent state can hold.

    Qdrant payloads contain untyped extras (None, lists, base64 blobs) that
    ``AgentState.documents`` cannot validate; keep only fields used downstream
    and expose the chunk text under 'content'.

    Args:
        payload: Raw point payload from Qdrant.

    Returns:
        A scalar-typed document dict for the agent state.
    """
    doc: dict[str, str | int | float | bool] = {
        "content": str(payload.get("text") or payload.get("content") or ""),
        "kb_id": str(payload.get("kb_id") or ""),
    }
    page_number = payload.get("page_number")
    if isinstance(page_number, int):
        doc["page_number"] = page_number
    image_url = payload.get("image_url")
    if isinstance(image_url, str):
        doc["image_url"] = image_url
    return doc


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

    # Embed the query with the same model used during ingestion
    query_text = cast(str, query)
    client = get_openai_client()
    response = await client.embeddings.create(
        input=query_text,
        model=settings.EMBEDDING_MODEL,
        encoding_format="float",
    )
    query_vector = response.data[0].embedding

    # Perform search
    # ponytail: search currently assumes 'knowledge_base' collection.
    results = await qdrant_manager.search(
        collection_name="knowledge_base",
        query_vector=query_vector,
        kb_id=kb_id,
        limit=5,
    )

    # Convert ScoredPoint results to sanitised metadata dicts for the state
    documents = [
        _sanitise_document(point.payload) for point in results if point.payload
    ]

    # Update and return state
    if isinstance(state, dict):
        state["documents"] = documents
    else:
        state.documents = documents

    return state
