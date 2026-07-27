from typing import Union

from openai import OpenAI
from src.schemas.agent_state import AgentState
from src.config.settings import get_settings


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
    # ponytail: defaults to OPENAI_API_KEY from environment via settings
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    if isinstance(state, dict):
        query = state.get("query", "")
    else:
        query = state.query

    prompt = (
        "You are a search query optimizer. Your goal is to rewrite the user's "
        "query into a concise, keyword-rich search term that will return the "
        "most relevant documents from a vector database.\n\n"
        f"Original Query: {query}\n\n"
        "Rewritten Query:"
    )

    response = client.chat.completions.create(
        model=getattr(settings, "REWRITER_MODEL_NAME", "gpt-4o"),
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
    else:
        state.rewritten_query = rewritten_query

    return state
