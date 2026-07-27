import json
from openai import OpenAI
from src.config.settings import get_settings
from src.schemas.agent_state import AgentState


async def grader_node(state: AgentState) -> AgentState:
    """Evaluates the relevance of retrieved documents to the query.

    Args:
        state (AgentState): The current state of the agent, including
            'query' and 'documents'.

    Returns:
        AgentState: The updated state with 'grader_feedback' populated.
    """
    settings = get_settings()

    # Initialize OpenAI client
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Prepare documents for prompt
    docs_text = "\n---\n".join([json.dumps(doc, indent=2) for doc in state.documents])

    # Evaluate relevance
    prompt = (
        "You are a grader evaluating the relevance of retrieved documents to a user query.\n"
        f"Query: {state.query}\n"
        f"Documents:\n{docs_text}\n\n"
        "Return 'relevant' if the documents contain information to answer the query, "
        "otherwise return 'not_relevant'. Return ONLY the status string."
    )

    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content
    feedback = content.strip().lower() if content else "not_relevant"

    # Normalize feedback
    if "relevant" in feedback and "not_relevant" not in feedback:
        feedback = "relevant"
    else:
        feedback = "not_relevant"

    # Update state
    # ponytail: we support both dict and AgentState object for LangGraph robustness.
    if isinstance(state, dict):
        state["grader_feedback"] = feedback
    else:
        state.grader_feedback = feedback

    return state
