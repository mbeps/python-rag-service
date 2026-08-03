from typing import Union

from src.schemas.agent_state import AgentState
from src.config.settings import get_settings
from src.utils.openai_client import get_openai_client


async def rewriter_node(state: Union[AgentState, dict]) -> Union[AgentState, dict]:
    """Reformulates the user query to better match document search patterns.

    This node uses an LLM to rewrite the query into a form more conducive
    to vector search retrieval, especially for complex or conversational queries.

    Args:
        state: The current graph state containing the original 'query'.

    Returns:
        The updated state with the rewritten query.
    """
    settings = get_settings()

    # Initialize OpenAI client
    client = get_openai_client()

    if isinstance(state, dict):
        query = state.get("rewritten_query") or state.get("query") or ""
        loop_step = state.get("loop_step", 0)
    else:
        query = state.rewritten_query or state.query
        loop_step = state.loop_step

    prompt = (
        "You are a search query optimizer. Your goal is to rewrite the user's "
        "query into a concise, keyword-rich search term that will return the "
        "most relevant documents from a vector database.\n\n"
        f"Query context: {query}\n\n"
        "Rewritten Query:"
    )

    response = await client.chat.completions.create(
        model=settings.GENERATION_MODEL,
        messages=[
            {"role": "system", "content": "You are a query rewriting assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    rewritten_query = content.strip() if content else query

    if isinstance(state, dict):
        state["rewritten_query"] = rewritten_query
        state["loop_step"] = loop_step + 1
    else:
        state.rewritten_query = rewritten_query
        state.loop_step = loop_step + 1

    return state
