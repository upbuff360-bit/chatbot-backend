from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

from openai import OpenAI

from app.chunking import chunk_text
from app.embedding_service import EmbeddingService
from app.manual_knowledge_service import ManualKnowledgeService
from app.pdf_service import PDFService
from app.prompt_builder import build_messages, expand_query, is_comparison_question, is_list_question
from app.vector_store import VectorStore
from app.website_service import WebsiteService

if TYPE_CHECKING:
    from app.services.chunk_store import ChunkStore

FALLBACK_ANSWER = "I don't have enough information to answer that."

# Phrases that indicate a fallback response — used for reliable fallback detection
# (Enhancement 8: broader than the previous hardcoded startswith check in chat.py)
FALLBACK_PHRASES = frozenset([
    "i don't have enough information",
    "i don't have details on that",
    "i don't have information",
    "i'm not sure about that",
    "i don't know",
    "that's outside my knowledge",
    "i cannot find",
    "i can't find",
    "no information available",
    "hmm, i don't",
])


def is_fallback_response(text: str) -> bool:
    """Return True if the response text indicates the bot couldn't answer."""
    lower = text.strip().lower()
    return any(phrase in lower for phrase in FALLBACK_PHRASES)


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

    # ── Ingestion ──────────────────────────────────────────────────────────────

    async def ingest_single_document(
        self,
        chunk_store: "ChunkStore",
        tenant_id: str,
        agent_id: str,
        document_id: str,
        source_type: str,
        source_name: str,
        text: str,
        category: str | None = None,
        page_title: str = "",
        page_url: str = "",
    ) -> int:
        """
        Ingest one document: save chunks to MongoDB then embed into Qdrant.

        Two-tier chunking strategy
        --------------------------
        For product and service pages (category == "product" | "service"):

          Tier 1 — Summary chunk (1 per page)
            A compact ~250-char string: "Product: <title>\\nURL: <url>\\n<first sentences>"
            Stored with chunk_type="summary" in both MongoDB and Qdrant payload.
            Matched first for broad listing queries ("what products do you offer?")
            so every product gets a slot regardless of top-k competition.

          Tier 2 — Detail chunks (N per page, normal 800-char splits)
            Standard sentence-boundary chunks used for specific follow-up
            questions ("tell me more about the CRM product").
            Stored with chunk_type="detail".

        For general/pricing pages only detail chunks are created (no summary).
        """
        if not text.strip() or not self.embedding_service.is_configured():
            return 0

        total_indexed = 0
        self.vector_store.initialize_collection(recreate=False)

        # ── Tier 1: Summary chunk (product and service pages only) ────────────
        if category in ("product", "service") and (page_title or page_url):
            from app.chunking import generate_summary_chunk
            summary_text = generate_summary_chunk(
                title=page_title,
                url=page_url,
                text=text,
            )
            if summary_text.strip():
                summary_ids = await chunk_store.save_chunks(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    document_id=document_id,
                    source_type=source_type,
                    source_name=source_name,
                    chunks=[summary_text],
                    category=category,
                    chunk_type="summary",
                )
                summary_embeddings = self.embedding_service.embed_texts([summary_text])
                self.vector_store.upsert_chunks(
                    chunks=[summary_text],
                    embeddings=summary_embeddings,
                    source_files=[source_name],
                    chunk_ids=summary_ids,
                    chunk_types=["summary"],
                )
                total_indexed += 1

        # ── Tier 2: Detail chunks (all pages) ─────────────────────────────────
        chunks = chunk_text(
            text,
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        )
        if not chunks:
            return total_indexed

        detail_ids = await chunk_store.save_chunks(
            tenant_id=tenant_id,
            agent_id=agent_id,
            document_id=document_id,
            source_type=source_type,
            source_name=source_name,
            chunks=chunks,
            category=category,
            chunk_type="detail",
        )

        detail_embeddings = self.embedding_service.embed_texts(chunks)
        self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=detail_embeddings,
            source_files=[source_name] * len(chunks),
            chunk_ids=detail_ids,
            chunk_types=["detail"] * len(chunks),
        )
        total_indexed += len(chunks)

        return total_indexed

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

        # Preserve chunk_type so summary chunks remain filterable after rebuild.
        # Old chunks without chunk_type default to "detail" — safe fallback.
        chunk_types  = [c.get("chunk_type", "detail") for c in all_chunks]

        embeddings = self.embedding_service.embed_texts(chunks_text)
        self.vector_store.upsert_chunks(
            chunks=chunks_text,
            embeddings=embeddings,
            source_files=source_files,
            chunk_ids=chunk_ids,
            chunk_types=chunk_types,
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
        self.vector_store.upsert_chunks(all_chunks, embeddings, source_files)
        return IngestionStats(documents_loaded=len(documents), chunks_indexed=len(all_chunks))

    def remove_document(self, doc_id: str, chunk_ids: list[str] | None = None) -> None:
        if chunk_ids:
            self.vector_store.delete_chunks_by_ids(chunk_ids)
        else:
            self.ingest_documents(recreate=True)

    # ── Vector search ──────────────────────────────────────────────────────────

    def search_chunks(self, question: str) -> tuple[list[str], list[dict]]:
        """
        Run vector search and return:
        - new_style_ids: chunk IDs to fetch from MongoDB
        - legacy_matches: chunks with text already in Qdrant payload (old data)

        Summary-chunk strategy (two-tier retrieval)
        -------------------------------------------
        For list questions ("what products do you offer?"), the pipeline first
        fetches every summary chunk (chunk_type="summary") via a Qdrant payload
        filter — no vector scoring needed.  These are prepended to new_style_ids
        so the LLM always sees a complete product list regardless of top-k.

        Normal vector search then fills the remaining slots with detail chunks
        for richer context on whichever products are most query-relevant.

        For non-list questions (specific product follow-ups, comparisons) the
        summary chunks are NOT prepended — only the standard vector search runs,
        so the answer stays focused on the asked product.
        """
        if not self.vector_store.collection_exists():
            return [], []

        # Enhancement 5: LLM-assisted query expansion
        query_variants = expand_query(question, llm_client=self.llm_client)
        all_matches: list = []
        seen_ids: set = set()

        if is_comparison_question(question):
            retrieval_limit = self.retrieval_limit * 2
            threshold = max(self.similarity_threshold - 0.05, 0.08)
        elif is_list_question(question):
            # Lower threshold so niche-term products (SAP, FSM, ERP) are not
            # silently dropped from listing results due to domain specificity.
            retrieval_limit = self.retrieval_limit * 3
            threshold = max(self.similarity_threshold - 0.10, 0.04)
        else:
            retrieval_limit = self.retrieval_limit
            threshold = self.similarity_threshold

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

        new_style_ids: list[str] = []
        legacy_matches: list[dict] = []

        for match in all_matches:
            payload = match.payload or {}
            if "chunk_id" in payload:
                new_style_ids.append(payload["chunk_id"])
            else:
                chunk = str(payload.get("chunk", "")).strip()
                source_file = str(payload.get("source_file", "unknown"))
                source_url  = str(payload.get("source_url", "")).strip()
                if chunk:
                    legacy_matches.append({
                        "content": chunk,
                        "source_name": source_url if source_url.startswith("http") else source_file,
                    })

        return new_style_ids, legacy_matches

    # ── Answer (sync, non-streaming) ───────────────────────────────────────────

    def answer_question(
        self,
        question: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        conversation_history: list[dict[str, str]] | None = None,
        prefetched_context: list[dict] | None = None,
    ) -> str:
        """
        Return a complete answer string (non-streaming).
        MongoDB fetching is done BEFORE calling this method (in the async route).
        """
        self._ensure_ready()

        context_parts, source_urls = self._build_context_parts(prefetched_context)

        # Enhancement 8: flag fallback BEFORE the LLM call so detection is 100% accurate
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
            # Enhancement (prev session): raised cap from 0.1 → 0.7 so the LLM
            # can phrase answers naturally instead of always picking the single
            # most probable token.
            temperature=min(temperature, 0.7),
        )
        return (response.choices[0].message.content or FALLBACK_ANSWER).strip()

    # ── Enhancement 3: Streaming answer ───────────────────────────────────────

    def stream_answer_question(
        self,
        question: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        conversation_history: list[dict[str, str]] | None = None,
        prefetched_context: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Yield answer tokens as they arrive from OpenAI (Server-Sent Events).
        The caller (FastAPI StreamingResponse) iterates this generator and
        pushes each chunk to the client immediately.

        Usage in a FastAPI route:
            from fastapi.responses import StreamingResponse

            def event_stream():
                for token in pipeline.stream_answer_question(...):
                    yield f"data: {token}\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        """
        self._ensure_ready()

        context_parts, source_urls = self._build_context_parts(prefetched_context)

        if not context_parts:
            yield FALLBACK_ANSWER
            return

        messages = build_messages(
            context="\n\n".join(context_parts),
            question=question,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            source_urls=source_urls,
        )

        stream = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=min(temperature, 0.7),
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_context_parts(
        self, prefetched_context: list[dict] | None
    ) -> tuple[list[str], list[str]]:
        """Deduplicate and format context chunks; return (context_parts, source_urls)."""
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

        return context_parts, source_urls

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