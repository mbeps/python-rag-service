import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.schemas.agent_state import AgentState


@pytest.mark.asyncio
async def test_retriever_node():
    """Tests the retriever_node for correct query selection and search call."""
    # Mock QdrantManager
    mock_qdrant_instance = MagicMock()
    # ScoredPoint objects should have payload
    mock_point = MagicMock()
    mock_point.payload = {"text": "found doc", "source": "test"}
    mock_qdrant_instance.search = AsyncMock(return_value=[mock_point])

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.KB_REGISTRY_COLLECTION = "knowledge_base"  # Default collection
    mock_settings.QDRANT_HOST = "localhost"
    mock_settings.QDRANT_PORT = 6333
    mock_settings.QDRANT_API_KEY = None

    # State with query and kb_id
    state = AgentState(query="original query", kb_id="test_kb", documents=[])

    # Patch the imports in the node module
    with (
        patch(
            "src.agent.nodes.retriever.QdrantManager", return_value=mock_qdrant_instance
        ),
        patch("src.agent.nodes.retriever.get_settings", return_value=mock_settings),
    ):
        # We need to import it inside the patch context if it performs actions on import,
        # but here we'll just import it normally at the top and patch the reference.
        from src.agent.nodes.retriever import retriever_node

        updated_state = await retriever_node(state)

    # Assertions
    assert len(updated_state.documents) == 1
    assert updated_state.documents[0]["text"] == "found doc"

    # Verify search call
    mock_qdrant_instance.search.assert_called_once()
    kwargs = mock_qdrant_instance.search.call_args.kwargs
    # Should check kwargs since they were called as keywords
    assert kwargs["collection_name"] == "knowledge_base"
    # kb_id should be passed
    assert kwargs["kb_id"] == "test_kb"
    # query should be original query
    assert kwargs["query"] == "original query"


@pytest.mark.asyncio
async def test_retriever_node_with_rewritten_query():
    """Tests that retriever_node prefers rewritten_query."""
    mock_qdrant_instance = MagicMock()
    mock_qdrant_instance.search = AsyncMock(return_value=[])
    mock_settings = MagicMock()
    mock_settings.KB_REGISTRY_COLLECTION = "knowledge_base"

    state = AgentState(
        query="original",
        rewritten_query="rewritten query",
        kb_id="test_kb",
        documents=[],
    )

    with (
        patch(
            "src.agent.nodes.retriever.QdrantManager", return_value=mock_qdrant_instance
        ),
        patch("src.agent.nodes.retriever.get_settings", return_value=mock_settings),
    ):
        from src.agent.nodes.retriever import retriever_node

        await retriever_node(state)

    # Verify search was called with rewritten query
    kwargs = mock_qdrant_instance.search.call_args.kwargs
    assert kwargs["query"] == "rewritten query"


@pytest.mark.asyncio
async def test_rewriter_node():
    """Tests the rewriter_node for correct query reformulation via LLM."""
    # Mock OpenAI client response
    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Better search query for documents"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    # Mock the chat.completions.create method
    mock_openai_client.chat.completions.create.return_value = mock_response

    # State with original query
    state = AgentState(query="Tell me about project alpha", kb_id="alpha_kb")

    # Patch the OpenAI client in the node module
    with patch("src.agent.nodes.rewriter.OpenAI", return_value=mock_openai_client):
        # We'll need to create this file
        from src.agent.nodes.rewriter import rewriter_node

        updated_state = await rewriter_node(state)

    # Assertions
    if isinstance(updated_state, AgentState):
        assert updated_state.rewritten_query == "Better search query for documents"
    else:
        assert updated_state["rewritten_query"] == "Better search query for documents"

    # Verify OpenAI call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "messages" in call_args
    assert any(
        "Tell me about project alpha" in msg["content"] for msg in call_args["messages"]
    )


@pytest.mark.asyncio
async def test_grader_node():
    """Tests the grader_node for correct relevance evaluation via LLM."""
    # Mock OpenAI client response
    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "relevant"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    # Mock the chat.completions.create method
    mock_openai_client.chat.completions.create.return_value = mock_response

    # State with query and some documents
    state = AgentState(
        query="What is the mission of project X?",
        kb_id="test_kb",
        documents=[{"text": "Project X mission is to build a RAG service."}],
    )

    # Patch the OpenAI client in the grader node module
    with (
        patch("src.agent.nodes.grader.OpenAI", return_value=mock_openai_client),
        patch("src.agent.nodes.grader.get_settings") as mock_get_settings,
    ):
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.GENERATION_MODEL = "gpt-4o"
        mock_settings.OPENAI_API_KEY = "test_key"
        mock_get_settings.return_value = mock_settings

        from src.agent.nodes.grader import grader_node

        updated_state = await grader_node(state)

    # Assertions
    assert updated_state.grader_feedback == "relevant"

    # Verify OpenAI call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "messages" in call_args
    # Check if prompt contains the documents and query
    prompt_content = next(
        msg["content"] for msg in call_args["messages"] if msg["role"] == "user"
    )
    assert "What is the mission of project X?" in prompt_content
    assert "Project X mission is to build a RAG service." in prompt_content


@pytest.mark.asyncio
async def test_generator_node():
    """Tests the generator_node for grounded answer and visual extraction."""
    # Mock OpenAI client response
    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Here is an answer with an image citation: ![Map](img_url)."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_openai_client.chat.completions.create.return_value = mock_response

    # State with query and documents (one with image_url)
    state = AgentState(
        query="Tell me about the campus map.",
        kb_id="test_kb",
        documents=[
            {
                "content": "The campus map shows building A.",
                "image_url": "img_url",
                "title": "Map",
            },
            {"content": "Building A is near the library.", "source": "docs"},
        ],
    )

    with (
        patch("src.agent.nodes.generator.OpenAI", return_value=mock_openai_client),
        patch("src.agent.nodes.generator.get_settings") as mock_get_settings,
    ):
        mock_settings = MagicMock()
        mock_settings.GENERATION_MODEL = "gpt-4o"
        mock_settings.OPENAI_API_KEY = "test_key"
        mock_get_settings.return_value = mock_settings

        from src.agent.nodes.generator import generator_node

        updated_state = await generator_node(state)

    # Assertions
    assert updated_state.answer is not None
    assert "image citation" in updated_state.answer
    # Verify visual_references extraction
    assert len(updated_state.visual_references) == 1
    assert updated_state.visual_references[0]["image_url"] == "img_url"
    assert updated_state.visual_references[0]["title"] == "Map"

    # Verify OpenAI call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args.kwargs
    prompt_content = next(
        msg["content"] for msg in call_args["messages"] if msg["role"] == "user"
    )
    assert "Tell me about the campus map." in prompt_content
    # Ensure BOTH documents were passed in context
    assert "The campus map shows building A." in prompt_content
    assert "Building A is near the library." in prompt_content
