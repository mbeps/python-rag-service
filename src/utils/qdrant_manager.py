from typing import Optional, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, ScoredPoint, VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
import asyncio

class QdrantManager:
    """
    Manager for Qdrant vector database operations using AsyncQdrantClient.
    """

    def __init__(self, client: AsyncQdrantClient):
        """
        Initializes the Qdrant manager with an existing async client.

        Args:
            client: An instance of AsyncQdrantClient.
        """
        self.client: AsyncQdrantClient = client

    async def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """
        Ensures the collection exists in Qdrant.
        """
        if await self.client.collection_exists(collection_name):
            info = await self.client.get_collection(collection_name)
            existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size == vector_size:
                return
            raise ValueError(
                f"Collection '{collection_name}' has vector size {existing_size}, "
                f"but {vector_size} is required."
            )

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    async def upsert_points(
        self, collection_name: str, points: List[PointStruct]
    ) -> None:
        """
        Upserts points to a collection with retries on transient errors.
        """
        max_retries = 3
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                await self.client.upsert(collection_name=collection_name, points=points)
                return
            except UnexpectedResponse as e:
                if (
                    isinstance(e.status_code, int)
                    and e.status_code >= 500
                    and attempt < max_retries - 1
                ):
                    await asyncio.sleep(retry_delay)
                    continue
                raise e

    async def search(
        self,
        collection_name: str,
        query_vector: Optional[List[float]] = None,
        query: Optional[str] = None,
        limit: int = 5,
        kb_id: Optional[str] = None,
    ) -> List[ScoredPoint]:
        """
        Performs similarity search, optionally filtered by kb_id.
        """
        target_query = query_vector if query_vector else query

        query_filter = None
        if kb_id:
            query_filter = Filter(
                must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
            )

        result = await self.client.query_points(
            collection_name=collection_name,
            query=target_query,
            query_filter=query_filter,
            limit=limit,
        )
        return result.points

    async def list_kbs(self, collection_name: str) -> List[dict]:
        """
        Lists all Knowledge Bases in the registry.
        """
        result, _ = await self.client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=100,
        )
        return [point.payload for point in result if point.payload] if result else []

    async def get_kb_metadata(self, collection_name: str, kb_id: str) -> Optional[dict]:
        """
        Retrieves metadata for a specific Knowledge Base.
        """
        result, _ = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if result and result[0].payload:
            return result[0].payload
        return None

    async def delete_points_by_filter(self, collection_name: str, kb_id: str, document_id: str) -> None:
        """
        Deletes points in a collection matching kb_id and document_id.
        """
        await self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="kb_id", match=MatchValue(value=kb_id)),
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                ]
            ),
        )

    async def delete_kb(self, collection_name: str, registry_collection: str, kb_id: str) -> None:
        """
        Deletes all points for a KB and its registry entry.
        """
        # Delete points in knowledge base collection
        await self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
            ),
        )

        # Delete registry entry
        await self.client.delete(
            collection_name=registry_collection,
            points_selector=Filter(
                must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
            ),
        )
