"""Factory for the OpenAI-compatible chat/embedding client."""

from openai import AsyncOpenAI

from src.config.settings import get_settings


def get_openai_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from settings; raises if the API key is missing."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set a valid key in your .env file (see .env.example)."
        )
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )
