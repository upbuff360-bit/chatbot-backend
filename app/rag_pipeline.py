from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from app.chunking import chunk_text
from app.embedding_service import EmbeddingService
from app.manual_knowledge_service import ManualKnowledgeService
from app.pdf_service import PDFService
from app.prompt_builder import build_messages, expand_query, is_comparison_question
from app.vector_store import VectorStore
from app.website_service import WebsiteService

if TYPE_CHECKING:
    from app.services.chunk_store import ChunkStore

FALLBACK_ANSWER = "I don't have enough information to answer that."


@dataclass(slots=True)
class IngestionStats:
    documents_loaded: int
    chunks_indexed: int


class RAGPipeline:
    def __init__(
        self,
        pdf_directory: str | Path,
        website_directory: str | Path | None = None,
        snippets_directory: str | Path | None = None,
        qa_directory: str | Path | None = None,
        collection_name: str = "knowledge_base",
    ) -> None:
        self.pdf_service = PDFService(pdf_directory=pdf_directory)
        self.website_service = WebsiteService(
            website_directory=website_directory or Path(pdf_directory).parent / "websites"
        )
        self.manual_knowledge_service = ManualKnowledgeService(
            snippets_directory=snippets_directory or Path(pdf_directory).parent / "text_snippets",
            qa_directory=qa_directory or Path(pdf_directory).parent / "qa",
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(collection_name=collection_name)
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
        self.relaxed_similarity_threshold = float(os.getenv("RELAXED_SIMILARITY_THRESHOLD", "0.08"))
        self.retrieval_limit = int(os.getenv("RETRIEVAL_LIMIT", "8"))
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.llm_client = OpenAI(api_key=api_key) if api_key else None

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest_single_document(
        self,
        chunk_store: "ChunkStore",
        tenant_id: str,
        agent_id: str,
        document_id: str,
        source_type: str,
        source_name: str,
        text: str,
    ) -> int:
        """Ingest one document: save chunks to MongoDB then embed into Qdrant."""
        if not text.strip() or not self.embedding_service.is_configured():
            return 0

        chunks = chunk_text(
            text,
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        )
        if not chunks:
            return 0

        # 1. Save text chunks to MongoDB → get stable IDs
        chunk_ids = await chunk_store.save_chunks(
            tenant_id=tenant_id,
            agent_id=agent_id,
            document_id=document_id,
            source_type=source_type,
            source_name=source_name,
            chunks=chunks,
        )

        # 2. Embed chunks
        embeddings = self.embedding_service.embed_texts(chunks)

        # 3. Store vectors in Qdrant (chunk_id in payload, no text)
        self.vector_store.initialize_collection(recreate=False)
        self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            source_files=[source_name] * len(chunks),
            chunk_ids=chunk_ids,
        )

        return len(chunks)

    async def rebuild_index_from_mongo(
        self,
        chunk_store: "ChunkStore",
        agent_id: str,
    ) -> int:
        """Rebuild Qdrant from MongoDB chunks — no re-upload needed."""
        all_chunks = await chunk_store.get_all_chunks_by_agent(agent_id)
        if not all_chunks:
            return 0

        self.vector_store.initialize_collection(recreate=True)
        chunks_text  = [c["content"] for c in all_chunks]
        chunk_ids    = [c["_id"] for c in all_chunks]
        source_files = [c["source_name"] for c in all_chunks]

        embeddings = self.embedding_service.embed_texts(chunks_text)
        self.vector_store.upsert_chunks(
            chunks=chunks_text,
            embeddings=embeddings,
            source_files=source_files,
            chunk_ids=chunk_ids,
        )
        return len(all_chunks)

    def ingest_documents(self, recreate: bool = True) -> IngestionStats:
        """Legacy disk-based ingestion — first-time setup only."""
        self.vector_store.initialize_collection(recreate=recreate)
        documents = self._load_documents()
        if not documents or not self.embedding_service.is_configured():
            return IngestionStats(documents_loaded=len(documents), chunks_indexed=0)

        all_chunks: list[str] = []
        source_files: list[str] = []
        for doc in documents:
            chunks = chunk_text(doc.text, chunk_size=800, overlap=150)
            all_chunks.extend(chunks)
            source_files.extend([doc.source_file] * len(chunks))

        if not all_chunks:
            return IngestionStats(documents_loaded=len(documents), chunks_indexed=0)

        embeddings = self.embedding_service.embed_texts(all_chunks)
        # Legacy: no chunk_ids, stores text in Qdrant payload
        self.vector_store.upsert_chunks(all_chunks, embeddings, source_files)
        return IngestionStats(documents_loaded=len(documents), chunks_indexed=len(all_chunks))

    def remove_document(self, doc_id: str, chunk_ids: list[str] | None = None) -> None:
        if chunk_ids:
            self.vector_store.delete_chunks_by_ids(chunk_ids)
        else:
            self.ingest_documents(recreate=True)

    # ── Vector search (sync, runs in executor) ────────────────────────────────

    def search_chunks(self, question: str) -> tuple[list[str], list[dict]]:
        """
        Run vector search and return:
        - new_style_ids: chunk IDs to fetch from MongoDB
        - legacy_matches: chunks with text already in Qdrant payload (old data)
        """
        if not self.vector_store.collection_exists():
            return [], []

        query_variants = expand_query(question)
        all_matches: list = []
        seen_ids: set = set()

        retrieval_limit = self.retrieval_limit * 2 if is_comparison_question(question) else self.retrieval_limit
        threshold = max(self.similarity_threshold - 0.05, 0.08) if is_comparison_question(question) else self.similarity_threshold

        for variant in query_variants:
            variant_embedding = self.embedding_service.embed_query(variant)
            for m in self.vector_store.search(
                query_embedding=variant_embedding,
                limit=retrieval_limit,
                score_threshold=threshold,
            ):
                mid = str(getattr(m, "id", None) or id(m))
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    all_matches.append(m)

        # Relaxed fallback
        if not all_matches:
            for variant in query_variants:
                variant_embedding = self.embedding_service.embed_query(variant)
                for m in self.vector_store.search(
                    query_embedding=variant_embedding,
                    limit=self.retrieval_limit,
                    score_threshold=0.0,
                ):
                    mid = str(getattr(m, "id", None) or id(m))
                    if float(m.score or 0.0) >= self.relaxed_similarity_threshold:
                        if mid not in seen_ids:
                            seen_ids.add(mid)
                            all_matches.append(m)

        if not all_matches:
            all_matches = self._keyword_matches(question)

        # Separate new-style vs legacy
        new_style_ids: list[str] = []
        legacy_matches: list[dict] = []

        for match in all_matches:
            payload = match.payload or {}
            if "chunk_id" in payload:
                new_style_ids.append(payload["chunk_id"])
            else:
                # Legacy: text stored in Qdrant payload
                chunk = str(payload.get("chunk", "")).strip()
                source_file = str(payload.get("source_file", "unknown"))
                source_url  = str(payload.get("source_url", "")).strip()
                if chunk:
                    legacy_matches.append({
                        "content": chunk,
                        "source_name": source_url if source_url.startswith("http") else source_file,
                    })

        return new_style_ids, legacy_matches

    # ── Answer (pure sync, no async calls) ───────────────────────────────────

    def answer_question(
        self,
        question: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        conversation_history: list[dict[str, str]] | None = None,
        prefetched_context: list[dict] | None = None,  # pre-fetched from MongoDB async
    ) -> str:
        """
        Answer using pre-fetched context.
        MongoDB fetching is done BEFORE calling this method (in the async route).
        This method is purely sync — safe to run in thread executor.
        """
        self._ensure_ready()

        context_parts: list[str] = []
        source_urls: list[str] = []
        seen_urls: set[str] = set()
        seen_chunks: set[str] = set()

        for item in (prefetched_context or []):
            content     = item.get("content", "").strip()
            source_name = item.get("source_name", "unknown")
            if content and content not in seen_chunks:
                seen_chunks.add(content)
                context_parts.append(f"[Source: {source_name}]\n{content}")
            if source_name.startswith("http") and source_name not in seen_urls:
                seen_urls.add(source_name)
                source_urls.append(source_name)

        if not context_parts:
            return FALLBACK_ANSWER

        messages = build_messages(
            context="\n\n".join(context_parts),
            question=question,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            source_urls=source_urls,
        )
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=min(temperature, 0.1),
        )
        return (response.choices[0].message.content or FALLBACK_ANSWER).strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        if not self.embedding_service.is_configured() or self.llm_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

    def _keyword_matches(self, question: str) -> list:
        question_terms = self._keyword_terms(question)
        if not question_terms:
            return []

        chunk_index: list[tuple[str, str, set[str]]] = []
        term_frequencies: Counter[str] = Counter()

        for doc in self._load_documents():
            for chunk in chunk_text(doc.text, chunk_size=800, overlap=150):
                chunk_terms = self._keyword_terms(chunk)
                if not chunk_terms:
                    continue
                chunk_index.append((chunk, doc.source_file, chunk_terms))
                term_frequencies.update(chunk_terms)

        matches: list[_KeywordMatch] = []
        for chunk, source_file, chunk_terms in chunk_index:
            overlap = question_terms & chunk_terms
            if not overlap:
                continue
            specificity = sum(1 / max(term_frequencies[t], 1) for t in overlap)
            coverage    = len(overlap) / len(question_terms)
            matches.append(_KeywordMatch(
                score=coverage + specificity,
                payload={"chunk": chunk, "source_file": source_file},
            ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[: self.retrieval_limit]

    def _load_documents(self) -> list:
        return [
            *self.pdf_service.load_documents(),
            *self.website_service.load_documents(),
            *self.manual_knowledge_service.load_documents(),
        ]

    @staticmethod
    def _keyword_terms(text: str) -> set[str]:
        return {
            t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text.lower())
            if t not in {
                "about", "document", "please", "summarize", "summary",
                "tell", "what", "which", "with", "this", "that", "have",
                "from", "into", "your", "there", "available",
            }
        }


@dataclass(slots=True)
class _KeywordMatch:
    score: float
    payload: dict[str, str]