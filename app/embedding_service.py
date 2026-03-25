from __future__ import annotations

import os

from openai import OpenAI


class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def is_configured(self) -> bool:
        return self.client is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._ensure_configured()
        if not texts:
            return []

        embeddings: list[list[float]] = []
        batch_size = 100
        for index in range(0, len(texts), batch_size):
            batch = texts[index:index + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        self._ensure_configured()
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding

    def _ensure_configured(self) -> None:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
