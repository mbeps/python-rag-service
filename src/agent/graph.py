from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes.generator import generator_node
from src.agent.nodes.grader import grader_node
from src.agent.nodes.retriever import retriever_node
from src.agent.nodes.rewriter import rewriter_node
from src.schemas.agent_state import AgentState


def decide_next_step(state: AgentState) -> Literal["generator", "rewriter"]:
    """Determines whether to proceed to generation or rewrite the query.

    Args:
        state (AgentState): The current state containing grader feedback.

    Returns:
        Literal["generator", "rewriter"]: The next node to execute.
    """
    if state.grader_feedback == "relevant":
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
        "rewriter": "rewriter",
    },
)

# Loop back rewriter to retriever
workflow.add_edge("rewriter", "retriever")

# End path
workflow.add_edge("generator", END)

# Checkpointer for state persistence
# ponytail: using MemorySaver as a simple in-memory checkpointer.
checkpointer = MemorySaver()

# Final app compilation
app = workflow.compile(checkpointer=checkpointer)
