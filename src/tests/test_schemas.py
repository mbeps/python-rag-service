from datetime import datetime
import pytest
from pydantic import ValidationError
from src.schemas.kb_metadata import KBMetadata
from src.schemas.ingestion_request import IngestionRequest
from src.schemas.query_request import QueryRequest
from src.schemas.query_response import QueryResponse
from src.schemas.agent_state import AgentState


def test_kb_metadata_valid():
    """Test KBMetadata with valid data."""
    data = {
        "kb_id": "test-kb",
        "name": "Test KB",
        "description": "A test knowledge base",
    }
    kb = KBMetadata(**data)
    assert kb.kb_id == data["kb_id"]
    assert isinstance(kb.created_at, datetime)


def test_kb_metadata_invalid():
    """Test KBMetadata with missing required fields."""
    with pytest.raises(ValidationError):
        KBMetadata.model_validate({"kb_id": "test-kb"})


def test_ingestion_request_valid():
    """Test IngestionRequest with valid data."""
    data = {
        "kb_id": "test-kb",
        "files": ["file1.pdf", "file2.txt"],
        "metadata": {"source": "web", "priority": 1},
    }
    req = IngestionRequest(**data)
    assert req.kb_id == data["kb_id"]
    assert req.files == data["files"]
    assert req.metadata is not None
    assert req.metadata["source"] == "web"


def test_ingestion_request_invalid():
    """Test IngestionRequest with invalid data types."""
    with pytest.raises(ValidationError):
        IngestionRequest(kb_id="test", files="not-a-list")


def test_query_request_valid():
    """Test QueryRequest with valid data and defaults."""
    req = QueryRequest(query="What is testing?")
    assert req.query == "What is testing?"
    assert req.kb_id is None
    assert req.use_multimodal is True


def test_query_request_full():
    """Test QueryRequest with all fields provided."""
    req = QueryRequest(query="test", kb_id="manual-kb", use_multimodal=False)
    assert req.kb_id == "manual-kb"
    assert req.use_multimodal is False


def test_query_response_valid():
    """Test QueryResponse with valid data."""
    data = {
        "answer": "This is a test.",
        "citations": [{"source": "doc1", "page": 1}],
        "visual_assets": ["img1.png"],
        "kb_id": "test-kb",
    }
    resp = QueryResponse(**data)
    assert resp.answer == data["answer"]
    assert resp.citations[0]["source"] == "doc1"


def test_query_response_invalid():
    """Test QueryResponse with missing fields."""
    with pytest.raises(ValidationError):
        QueryResponse.model_validate({"answer": "test", "kb_id": "test"})


def test_agent_state_valid():
    """Test AgentState with valid data."""
    data = {"query": "hello", "kb_id": "test-kb"}
    state = AgentState(**data)
    assert state.query == "hello"
    assert state.documents == []
    assert state.answer is None


def test_agent_state_full():
    """Test AgentState with full data."""
    data = {
        "query": "hello",
        "kb_id": "test-kb",
        "documents": [{"content": "text", "score": 0.9}],
        "answer": "world",
        "rewritten_query": "hello rewritten",
        "grader_feedback": "good",
        "visual_references": [{"id": "v1"}],
    }
    state = AgentState(**data)
    assert state.answer == "world"
    assert state.documents[0]["content"] == "text"
    assert state.visual_references[0]["id"] == "v1"
