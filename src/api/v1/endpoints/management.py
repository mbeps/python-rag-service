from typing import List
from fastapi import APIRouter, HTTPException, Depends
from src.config.settings import get_settings, Settings
from src.utils.qdrant_manager import QdrantManager
from src.schemas.kb_metadata import KBMetadata

router = APIRouter()


async def get_qdrant_manager(
    settings: Settings = Depends(get_settings),
) -> QdrantManager:
    """
    Dependency to provide a QdrantManager instance.

    Args:
        settings: Application settings.

    Returns:
        An initialized QdrantManager.
    """
    return QdrantManager(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )


@router.get("/kb", response_model=List[KBMetadata])
async def list_knowledge_bases(
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
) -> List[KBMetadata]:
    """
    List all Knowledge Bases stored in the registry.

    Args:
        settings: Application settings.
        qdrant: Qdrant manager.

    Returns:
        List of Knowledge Base metadata.
    """
    kbs = await qdrant.list_kbs(settings.KB_REGISTRY_COLLECTION)
    return [KBMetadata(**kb) for kb in kbs]


@router.get("/kb/{kb_id}", response_model=KBMetadata)
async def get_knowledge_base_metadata(
    kb_id: str,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
) -> KBMetadata:
    """
    Get metadata for a specific Knowledge Base.

    Args:
        kb_id: The unique identifier for the Knowledge Base.
        settings: Application settings.
        qdrant: Qdrant manager.

    Returns:
        The Knowledge Base metadata.

    Raises:
        HTTPException: If the Knowledge Base is not found.
    """
    kb = await qdrant.get_kb_metadata(settings.KB_REGISTRY_COLLECTION, kb_id)
    if not kb:
        raise HTTPException(
            status_code=404, detail=f"Knowledge Base with ID '{kb_id}' not found."
        )
    return KBMetadata(**kb)
