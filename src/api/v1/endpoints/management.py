from typing import List
from fastapi import APIRouter, HTTPException, Depends, Response
from src.config.settings import get_settings, Settings
from src.utils.qdrant_manager import QdrantManager
from src.utils.minio_manager import MinIOManager
from src.schemas.kb_metadata import KBMetadata
from src.api.v1.dependencies import get_qdrant_manager, get_minio_manager

router = APIRouter()


@router.get("/kb", response_model=List[KBMetadata])
async def list_knowledge_bases(
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
) -> List[KBMetadata]:
    """
    List all Knowledge Bases stored in the registry.
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
    """
    kb = await qdrant.get_kb_metadata(settings.KB_REGISTRY_COLLECTION, kb_id)
    if not kb:
        raise HTTPException(
            status_code=404, detail=f"Knowledge Base with ID '{kb_id}' not found."
        )
    return KBMetadata(**kb)


@router.delete("/kb/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: str,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    minio: MinIOManager = Depends(get_minio_manager),
):
    """
    Delete a Knowledge Base and all its associated data.
    """
    # Check if exists first
    kb = await qdrant.get_kb_metadata(settings.KB_REGISTRY_COLLECTION, kb_id)
    if not kb:
        raise HTTPException(
            status_code=404, detail=f"Knowledge Base with ID '{kb_id}' not found."
        )

    # 1. Delete points in Qdrant and registry entry
    await qdrant.delete_kb(
        collection_name=settings.QDRANT_COLLECTION,
        registry_collection=settings.KB_REGISTRY_COLLECTION,
        kb_id=kb_id,
    )

    # 2. Delete MinIO folder
    await minio.delete_folder(
        bucket_name=settings.MINIO_VISUAL_BUCKET, prefix=f"{kb_id}/"
    )

    return Response(status_code=204)


@router.delete("/kb/{kb_id}/documents/{document_id}", status_code=204)
async def delete_document(
    kb_id: str,
    document_id: str,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
):
    """
    Surgically remove a document from a Knowledge Base.
    """
    # We don't check existence separately for performance,
    # delete_points_by_filter will just not delete anything if not found.
    # However, we might want to check if the KB exists.
    kb = await qdrant.get_kb_metadata(settings.KB_REGISTRY_COLLECTION, kb_id)
    if not kb:
        raise HTTPException(
            status_code=404, detail=f"Knowledge Base with ID '{kb_id}' not found."
        )

    await qdrant.delete_points_by_filter(
        collection_name=settings.QDRANT_COLLECTION,
        kb_id=kb_id,
        document_id=document_id,
    )

    return Response(status_code=204)
