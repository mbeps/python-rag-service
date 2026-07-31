# AI RAG Knowledge Base Service

This project is a RAG knowledge base service designed to turn documents into a searchable, context-aware information layer. It helps users ask questions over collected content and receive answers grounded in relevant retrieved material. The service is built to support document-aware workflows with a simple and practical retrieval experience. It provides a solid foundation for exploring and querying knowledge bases with clear, grounded responses.

# Features

- Ingests documents into a searchable knowledge base for retrieval-based question answering
- Supports grounded responses by returning answers tied to relevant retrieved content
- Handles document-aware workflows with a simple query experience
- Provides a practical foundation for building and exploring knowledge-base assistants

# Stack

- [FastAPI](https://fastapi.tiangolo.com/): Provides the web API layer for serving requests and managing ingestion workflows.
- [LangGraph](https://langchain-ai.github.io/langgraph/): Orchestrates the retrieval and generation flow as a stateful agentic process.
- [Qdrant](https://qdrant.tech/): Stores and searches vector embeddings for knowledge-base retrieval.
- [MinIO](https://min.io/): Stores uploaded documents and extracted assets in an S3-compatible object store.

# Requirements

- Python 3.12 or newer
- uv for dependency management and running the project
- Docker to run the required containers unless they are already hosted
- Access to an OpenAI-compatible API provider for generation and embedding tasks

# Setting Up Environment

## Environment Variables

Create a `.env` file in the project root by copying `.env.example` and filling in your credentials:

```sh
cp .env.example .env
```

The service uses any OpenAI-compatible API for generation and embeddings. Configure the following variables:

### API Configuration

- **`OPENAI_BASE_URL`**: Base URL for your OpenAI-compatible API provider (e.g., `https://openrouter.ai/api/v1`, `https://api.openai.com/v1`). This allows flexibility to use OpenRouter, Azure OpenAI, local inference servers, or other compatible APIs.
- **`OPENAI_API_KEY`**: API key for authenticating with your chosen provider. Obtain this from your API provider's dashboard.
- **`GENERATION_MODEL`**: Model identifier for response generation (e.g., `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, `gpt-4o-mini`). Different providers offer different models; check your provider's documentation.
- **`EMBEDDING_MODEL`**: Model identifier for creating vector embeddings from text (e.g., `nvidia/nemotron-3-embed-1b:free`, `text-embedding-3-small`).
- **`EMBEDDING_DIMENSIONS`**: Dimensionality of the embeddings produced by your embedding model (e.g., `2048`, `1536`). This must match the model's output dimensions.

### Search Configuration

- **`KB_MATCH_THRESHOLD`**: Minimum similarity score (0.0–1.0) for search results to be considered relevant. Higher values (e.g., `0.75`) return only highly similar results; lower values (e.g., `0.5`) are more lenient. Default: `0.65`.

### Qdrant (Vector Database)

- **`QDRANT_HOST`**: Hostname or IP of the Qdrant server (e.g., `localhost`, `qdrant.example.com`).
- **`QDRANT_PORT`**: Port on which Qdrant listens (default: `6333`).

### MinIO (Object Storage)

- **`MINIO_ENDPOINT`**: Endpoint of the MinIO server, including port (e.g., `localhost:9000`, `minio.example.com:9000`).
- **`MINIO_ACCESS_KEY`**: Access key for MinIO authentication. Use a strong, unique value in production.
- **`MINIO_SECRET_KEY`**: Secret key for MinIO authentication. Use a strong, unique value in production.
- **`MINIO_BUCKET`**: Name of the S3-compatible bucket where uploaded documents and assets are stored (e.g., `rag-assets`). The service will create this bucket if it doesn't exist.

# Running Service

1. Start the required containers for Qdrant and MinIO using Docker Compose.
```sh
docker compose up -d qdrant minio
```

2. Running the service with uvicorn:
```sh
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Available Endpoints

The service runs at `http://localhost:8000` and exposes interactive API documentation at `http://localhost:8000/docs`.

### Health Check
- **Method:** `GET`
- **Endpoint:** `/health`
- **Description:** Confirms that the service is running.
- **Example:**
```sh
curl http://localhost:8000/health
```

### Ingest Documents
- **Method:** `POST`
- **Endpoint:** `/api/v1/ingest`
- **Description:** Uploads one or more files into a knowledge base in the background.
- **Example:**
```sh
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "kb_id=finance_docs" \
  -F "kb_name=Financial Statements" \
  -F "kb_description=Annual reports and financial documents" \
  -F "files=@path/to/document.pdf"
```

### Query Knowledge Base
- **Method:** `POST`
- **Endpoint:** `/api/v1/query`
- **Description:** Sends a question to a specific or dynamically selected knowledge base.
- **Example:**
```sh
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key financial metrics?",
    "kb_id": "finance_docs",
    "use_multimodal": true
  }'
```

### List Knowledge Bases
- **Method:** `GET`
- **Endpoint:** `/api/v1/kb`
- **Description:** Returns metadata for all available knowledge bases.
- **Example:**
```sh
curl http://localhost:8000/api/v1/kb
```

### Get Knowledge Base Metadata
- **Method:** `GET`
- **Endpoint:** `/api/v1/kb/{kb_id}`
- **Description:** Returns metadata for one specific knowledge base.
- **Example:**
```sh
curl http://localhost:8000/api/v1/kb/finance_docs
```

# References

- [Python Documentation](https://www.python.org/doc/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [MinIO Documentation](https://min.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)