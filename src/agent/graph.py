from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.generator import generator_node
from src.agent.nodes.grader import grader_node
from src.agent.nodes.retriever import retriever_node
from src.agent.nodes.rewriter import rewriter_node
from src.schemas.agent_state import AgentState


def decide_next_step(state: AgentState) -> Literal["generator", "rewriter", "generate"]:
    """Determines whether to proceed to generation or rewrite the query.

    Args:
        state (AgentState): The current state containing grader feedback and loop counter.

    Returns:
        Literal["generator", "rewriter", "generate"]: The next node to execute.
    """
    # ponytail: supporting both dict and object access for LangGraph state robustness.
    loop_step = state["loop_step"] if isinstance(state, dict) else state.loop_step

    if loop_step >= 3:
        # ponytail: cap iterations at 3 to prevent infinite loops.
        return "generate"

    grader_feedback = (
        state["grader_feedback"] if isinstance(state, dict) else state.grader_feedback
    )
    if grader_feedback == "relevant":
        return "generator"
    return "rewriter"


# Initialize the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retriever", retriever_node)
workflow.add_node("grader", grader_node)
workflow.add_node("rewriter", rewriter_node)
workflow.add_node("generator", generator_node)

# Define edges
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "grader")

# Add conditional edge
workflow.add_conditional_edges(
    "grader",
    decide_next_step,
    {
        "generator": "generator",
        "generate": "generator",
        "rewriter": "rewriter",
    },
)

# Loop back rewriter to retriever
workflow.add_edge("rewriter", "retriever")

# End path
workflow.add_edge("generator", END)

# Final app compilation
# ponytail: Removed MemorySaver to prevent memory leaks in production (H7).
# For conversational memory, a database-backed checkpointer should be used.
app = workflow.compile()
