import pytest
from pydantic import ValidationError
from src.config.settings import Settings


def test_settings_defaults(monkeypatch):
    """Test that settings have correct default values."""
    # Ensure environment variables don't interfere with default tests
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    # We also need to bypass loading from .env for the default test
    # One way is to create a Settings instance with a non-existent env_file
    # or just assume the env/settings are clean if we monkeypatch them out.
    settings = Settings(_env_file=None)
    assert settings.APP_NAME == "Agentic RAG Service"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.OPENAI_API_KEY is None
    assert settings.OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.GENERATION_MODEL == "gpt-4o-mini"
    assert settings.QDRANT_HOST == "localhost"
    assert settings.QDRANT_PORT == 6333
    assert settings.QDRANT_API_KEY is None
    assert settings.MINIO_ENDPOINT == "localhost:9000"
    assert settings.MINIO_ACCESS_KEY == "minioadmin"
    assert settings.MINIO_SECRET_KEY == "minioadmin"
    assert settings.MINIO_SECURE is False
    assert settings.KB_REGISTRY_COLLECTION == "kb_registry"
    assert settings.DYNAMIC_KB_THRESHOLD == 0.65


def test_settings_env_overrides(monkeypatch):
    """Test that environment variables correctly override default settings."""
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("QDRANT_PORT", "1234")
    monkeypatch.setenv("MINIO_SECURE", "True")
    monkeypatch.setenv("DYNAMIC_KB_THRESHOLD", "0.8")

    settings = Settings()
    assert settings.APP_NAME == "Test App"
    assert settings.QDRANT_PORT == 1234
    assert settings.MINIO_SECURE is True
    assert settings.DYNAMIC_KB_THRESHOLD == 0.8


def test_settings_invalid_types(monkeypatch):
    """Test that invalid types for settings raise a ValidationError."""
    monkeypatch.setenv("QDRANT_PORT", "not-an-int")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_invalid_float(monkeypatch):
    """Test that invalid float for threshold raises a ValidationError."""
    monkeypatch.setenv("DYNAMIC_KB_THRESHOLD", "not-a-float")
    with pytest.raises(ValidationError):
        Settings()


def test_openai_api_key_stripped(monkeypatch):
    """Test that OPENAI_API_KEY is stripped of whitespace and surrounding quotes."""
    monkeypatch.setenv("OPENAI_API_KEY", '  "sk-or-v1-abc"  ')
    settings = Settings(_env_file=None)
    assert settings.OPENAI_API_KEY == "sk-or-v1-abc"


def test_openai_api_key_blank_becomes_none(monkeypatch):
    """Test that an empty OPENAI_API_KEY becomes None."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    settings = Settings(_env_file=None)
    assert settings.OPENAI_API_KEY is None
