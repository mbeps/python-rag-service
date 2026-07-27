from fastapi import APIRouter

from src.api.v1.endpoints import ingestion, query, management

api_router = APIRouter()
api_router.include_router(ingestion.router, tags=["ingestion"])
api_router.include_router(query.router, tags=["query"])
api_router.include_router(management.router, tags=["management"])
