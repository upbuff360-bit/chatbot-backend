from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorStore:
    _shared_client: QdrantClient | None = None
    _shared_client_key: tuple[str, str] | None = None

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        vector_size: int = 1536,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.client = self._build_client()

    def initialize_collection(self, recreate: bool = False) -> None:
        if recreate and self._collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source_files: list[str],
        chunk_ids: list[str] | None = None,
    ) -> None:
        """
        Upsert chunks into Qdrant.
        If chunk_ids provided (MongoDB IDs), store them in payload instead of full text.
        This makes Qdrant a pure search index — text lives in MongoDB.
        """
        if not chunks:
            return

        points = []
        for i, (chunk, embedding, source_file) in enumerate(
            zip(chunks, embeddings, source_files, strict=True)
        ):
            point_id = chunk_ids[i] if chunk_ids else str(uuid4())

            # If chunk_ids provided → Qdrant is ID-only index (text in MongoDB)
            # If no chunk_ids → legacy mode, store chunk text in payload
            payload = {
                "source_file": source_file,
                "source_url": source_file if source_file.startswith("http") else "",
            }
            if chunk_ids:
                payload["chunk_id"] = chunk_ids[i]
            else:
                payload["chunk"] = chunk  # legacy fallback

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> None:
        """Delete specific points from Qdrant by their IDs."""
        if not chunk_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=chunk_ids),
        )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        score_threshold: float = 0.75,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return response.points

    def collection_exists(self) -> bool:
        return self._collection_exists()

    def delete_collection(self) -> None:
        if self._collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    @classmethod
    def close_shared_client(cls) -> None:
        if cls._shared_client is not None:
            cls._shared_client.close()
            cls._shared_client = None
            cls._shared_client_key = None

    @classmethod
    def _build_client(cls) -> QdrantClient:
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

        if qdrant_url:
            return QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)

        qdrant_path = Path(os.getenv("QDRANT_PATH", "./persist/qdrant")).resolve()
        qdrant_path.mkdir(parents=True, exist_ok=True)
        client_key = ("local", str(qdrant_path))
        if cls._shared_client is not None and cls._shared_client_key == client_key:
            return cls._shared_client

        cls._shared_client = QdrantClient(path=str(qdrant_path))
        cls._shared_client_key = client_key
        return cls._shared_client