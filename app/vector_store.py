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
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                )
            except Exception as exc:
                # 409 Conflict means collection was created between the exists-check
                # and the create call (race condition). Safe to ignore.
                if "already exists" in str(exc).lower() or "409" in str(exc):
                    pass
                else:
                    raise

    def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source_files: list[str],
        chunk_ids: list[str] | None = None,
        chunk_types: list[str] | None = None,
    ) -> None:
        """
        Upsert chunks into Qdrant.

        chunk_ids   — if provided, Qdrant stores only the ID; full text lives
                      in MongoDB (preferred mode).
        chunk_types — parallel list of "summary" | "detail" per chunk.
                      Stored in the Qdrant payload so get_summary_chunk_ids()
                      can filter without touching MongoDB.
        """
        if not chunks:
            return

        points = []
        for i, (chunk, embedding, source_file) in enumerate(
            zip(chunks, embeddings, source_files, strict=True)
        ):
            point_id = chunk_ids[i] if chunk_ids else str(uuid4())

            payload = {
                "source_file": source_file,
                "source_url": source_file if source_file.startswith("http") else "",
            }
            if chunk_ids:
                payload["chunk_id"] = chunk_ids[i]
            else:
                payload["chunk"] = chunk  # legacy fallback

            # Store chunk_type so summary chunks can be filtered via Qdrant
            # scroll without a MongoDB round-trip.
            if chunk_types:
                payload["chunk_type"] = chunk_types[i]
            else:
                payload["chunk_type"] = "detail"  # safe default for old data

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def get_summary_chunk_ids(self) -> list[str]:
        """
        Scroll the Qdrant collection and return chunk_ids for every point
        whose payload has chunk_type == "summary".

        Called synchronously from search_chunks() for list questions — no
        MongoDB round-trip required because the chunk_type is stored in the
        Qdrant payload at ingestion time.
        """
        result_ids: list[str] = []
        offset = None

        while True:
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_type",
                            match=models.MatchValue(value="summary"),
                        )
                    ]
                ),
                limit=200,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            points, next_offset = response
            for p in points:
                cid = (p.payload or {}).get("chunk_id")
                if cid:
                    result_ids.append(cid)
            if next_offset is None:
                break
            offset = next_offset

        return result_ids

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