from src.config.settings import get_settings
from src.schemas.agent_state import AgentState
from src.utils.openai_client import get_openai_client


async def generator_node(state: AgentState) -> AgentState:
    """Synthesises a grounded answer using retrieved documents and extracts visual references.

    Args:
        state (AgentState): The current state of the agent, including
            'query' and 'documents'.

    Returns:
        AgentState: The updated state with 'answer' populated and
            'visual_references' extracted from document metadata.
    """
    settings = get_settings()

    client = get_openai_client()

    # 1. Extract visual references
    # Iterate through documents and look for 'image_url'
    visual_refs = []
    for doc in state.documents:
        if isinstance(doc, dict) and "image_url" in doc:
            visual_refs.append(doc)

    state.visual_references = visual_refs

    # 2. Prepare context for synthesis
    # ponytail: joining document contents with double newlines
    context_str = "\n\n".join(
        [
            str(doc.get("content", ""))
            for doc in state.documents
            if isinstance(doc, dict)
        ]
    )

    # 3. Define prompts
    system_prompt = (
        "You are a helpful AI assistant. Answer the user question based ONLY on the provided context. "
        "If the context contains an 'image_url' for a document, use it to provide an inline visual citation "
        "in Markdown format: ![Description](image_url). Ensure your answer is concise and grounded."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {state.query}"

    # 4. Generate answer
    response = await client.chat.completions.create(
        model=settings.GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    state.answer = response.choices[0].message.content
    return state
