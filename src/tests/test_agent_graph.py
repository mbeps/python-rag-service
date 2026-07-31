import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agent.graph import app


@pytest.fixture
def mock_retriever():
    with (
        patch("src.agent.nodes.retriever.QdrantManager", autospec=True) as mock,
        patch("src.agent.nodes.retriever.get_openai_client") as mock_client_factory,
    ):
        mock_inst = mock.return_value
        mock_inst.search = AsyncMock(
            return_value=[
                MagicMock(
                    payload={"content": "Context info", "image_url": "http://img.png"}
                )
            ]
        )
        mock_client = mock_client_factory.return_value
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 8
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [mock_embedding]
        mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        yield mock_inst


@pytest.fixture
def mock_grader():
    with patch("src.agent.nodes.grader.get_openai_client") as mock_factory:
        mock_inst = mock_factory.return_value
        mock_cmpl = MagicMock()
        mock_cmpl.create = AsyncMock()
        mock_inst.chat.completions = mock_cmpl
        yield mock_cmpl


@pytest.fixture
def mock_rewriter():
    with patch("src.agent.nodes.rewriter.get_openai_client") as mock_factory:
        mock_inst = mock_factory.return_value
        mock_cmpl = MagicMock()
        mock_cmpl.create = AsyncMock()
        mock_inst.chat.completions = mock_cmpl
        yield mock_cmpl


@pytest.fixture
def mock_generator():
    with patch("src.agent.nodes.generator.get_openai_client") as mock_factory:
        mock_inst = mock_factory.return_value
        mock_cmpl = MagicMock()
        mock_cmpl.create = AsyncMock()
        mock_inst.chat.completions = mock_cmpl
        yield mock_cmpl


@pytest.mark.asyncio
async def test_agent_workflow(mock_retriever, mock_grader, mock_generator):
    """E2E test for the agent graph workflow - Success Path."""

    # Success path: relevant feedback
    mock_grader.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="relevant"))]
    )

    # Generation
    mock_generator.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Final Answer"))]
    )

    initial_state = {
        "query": "What is Python?",
        "kb_id": "test_kb",
        "documents": [],
        "visual_references": [],
    }

    config = {"configurable": {"thread_id": "test_thread"}}
    result = await app.ainvoke(initial_state, config=config)

    assert "Final Answer" in result["answer"]
    assert result["grader_feedback"] == "relevant"
    assert len(result["documents"]) == 1
    assert len(result["visual_references"]) == 1


@pytest.mark.asyncio
async def test_agent_workflow_relevance_loop(
    mock_retriever, mock_grader, mock_rewriter, mock_generator
):
    """E2E test for the agent graph workflow - Loop Path."""

    # Grader: not_relevant then relevant
    mock_grader.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="not_relevant"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="relevant"))]),
    ]

    # Rewriter
    mock_rewriter.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="optimized query"))]
    )

    # Generator
    mock_generator.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Loop Answer"))]
    )

    initial_state = {
        "query": "Complex query",
        "kb_id": "test_kb",
        "documents": [],
        "visual_references": [],
    }

    config = {"configurable": {"thread_id": "loop_thread"}}
    result = await app.ainvoke(initial_state, config=config)

    assert "Loop Answer" in result["answer"]
    assert mock_retriever.search.call_count == 2
    assert mock_rewriter.create.call_count == 1
