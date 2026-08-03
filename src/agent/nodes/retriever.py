from typing import cast
from qdrant_client import AsyncQdrantClient
from src.config.settings import get_settings, settings
from src.schemas.agent_state import AgentState
from src.utils.openai_client import get_openai_client
from src.utils.qdrant_manager import QdrantManager


def get_qdrant_client() -> AsyncQdrantClient:
    """Creates a temporary async client for the node."""
    return AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )


def _sanitise_document(
    payload: dict[str, object],
) -> dict[str, str | int | float | bool]:
    """Map a Qdrant payload to the fields the agent state can hold."""
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
    """Retrieves relevant documents from the knowledge base."""
    settings = get_settings()

    # ponytail: The node currently creates its own client; in production,
    # this should be passed in via RunnableConfig if possible.
    client = get_qdrant_client()
    qdrant_manager = QdrantManager(client=client)

    try:
        if isinstance(state, dict):
            query = state.get("rewritten_query") or state.get("query")
            kb_id = state.get("kb_id")
        else:
            query = state.rewritten_query or state.query
            kb_id = state.kb_id

        query_text = cast(str, query)
        openai_client = get_openai_client()
        response = await openai_client.embeddings.create(
            input=query_text,
            model=settings.EMBEDDING_MODEL,
            encoding_format="float",
        )
        query_vector = response.data[0].embedding

        results = await qdrant_manager.search(
            collection_name="knowledge_base",
            query_vector=query_vector,
            kb_id=kb_id,
            limit=5,
        )

        documents = [
            _sanitise_document(point.payload) for point in results if point.payload
        ]

        if isinstance(state, dict):
            state["documents"] = documents
        else:
            state.documents = documents

        return state
    finally:
        await client.close()
