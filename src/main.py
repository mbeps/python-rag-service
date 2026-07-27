"""Agentic RAG Service entry point."""

from fastapi import FastAPI

app = FastAPI(title="Agentic RAG Service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
