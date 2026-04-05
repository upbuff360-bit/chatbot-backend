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
from app.prompt_builder import (
    build_messages,
    build_required_career_followup,
    build_required_lead_capture_followup,
    expand_query,
    extract_comparison_items,
    is_comparison_question,
    is_list_question,
)
from app.vector_store import VectorStore
from app.website_service import WebsiteService

if TYPE_CHECKING:
    from app.services.chunk_store import ChunkStore

FALLBACK_ANSWER = (
    "Thank you for your question. At the moment, I do not have the relevant information available in my current context. "
    "If you would like, please let me know if there is anything else I can help you with."
)

# Phrases that indicate a fallback response — used for reliable fallback detection
# (Enhancement 8: broader than the previous hardcoded startswith check in chat.py)
FALLBACK_PHRASES = frozenset([
    "i don't have enough information",
    "i don't have details on that",
    "i don't have information",
    "i don't have that detail",
    "i don't have that information",
    "i'm not sure about that",
    "i'd love to help, but i don't have",
    "thank you for your question. at the moment, i do not have",
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
        self.retrieval_limit = int(os.getenv("RETRIEVAL_LIMIT", "5"))
        self.max_context_chunks = int(os.getenv("CONTEXT_MAX_CHUNKS", "8"))
        self.max_context_chars = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))
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
        catalog_categories: list[str] | None = None,
    ) -> int:
        """
        Ingest one document: save chunks to MongoDB then embed into Qdrant.

        Two-tier chunking strategy
        --------------------------
        For product, service and pricing pages (category == "product" | "service" | "pricing"):

          Tier 1 — Summary chunk (1 per page)
            A compact ~250-char string: "Product: <title>\\nURL: <url>\\n<first sentences>"
            Stored with chunk_type="summary" in both MongoDB and Qdrant payload.
            Matched first for broad listing queries ("what products do you offer?")
            so every product gets a slot regardless of top-k competition.

          Tier 2 — Detail chunks (N per page, normal 800-char splits)
            Standard sentence-boundary chunks used for specific follow-up
            questions ("tell me more about the CRM product").
            Stored with chunk_type="detail".

        For general pages only detail chunks are created (no summary).
        """
        if not text.strip() or not self.embedding_service.is_configured():
            return 0

        total_indexed = 0
        self.vector_store.initialize_collection(recreate=False)

        # ── Tier 1: Summary chunk (product, service and pricing pages only) ─────
        if category in ("product", "service", "pricing"):
            from app.chunking import generate_catalog_summary_chunks, generate_summary_chunk
            summary_texts: list[str] = []
            if page_title or page_url:
                summary_text = generate_summary_chunk(
                    title=page_title,
                    url=page_url,
                    text=text,
                    category=category,
                    catalog_categories=catalog_categories,
                )
                if summary_text.strip():
                    summary_texts.append(summary_text)
            else:
                summary_texts = generate_catalog_summary_chunks(
                    text=text,
                    category=category,
                )
                # Fallback: if the document structure isn't clear enough for
                # section extraction (e.g. a flat service description file),
                # create one summary chunk using the filename as the title so
                # the document still appears in listing answers.
                if not summary_texts and source_name:
                    from pathlib import Path as _Path
                    fallback_title = (
                        _Path(source_name).stem
                        .replace("_", " ")
                        .replace("-", " ")
                        .strip()
                    )
                    if fallback_title:
                        fallback_text = generate_summary_chunk(
                            title=fallback_title,
                            url="",
                            text=text,
                            category=category,
                        )
                        if fallback_text.strip():
                            summary_texts = [fallback_text]
            if summary_texts:
                summary_ids = await chunk_store.save_chunks(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    document_id=document_id,
                    source_type=source_type,
                    source_name=source_name,
                    chunks=summary_texts,
                    category=category,
                    chunk_type="summary",
                )
                summary_embeddings = self.embedding_service.embed_texts(summary_texts)
                self.vector_store.upsert_chunks(
                    chunks=summary_texts,
                    embeddings=summary_embeddings,
                    source_files=[source_name] * len(summary_texts),
                    chunk_ids=summary_ids,
                    chunk_types=["summary"] * len(summary_texts),
                )
                total_indexed += len(summary_texts)

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

        def _search_variants(variants: list[str], *, limit: int, score_threshold: float) -> None:
            for variant in variants:
                variant_embedding = self.embedding_service.embed_query(variant)
                for m in self.vector_store.search(
                    query_embedding=variant_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                ):
                    mid = str(getattr(m, "id", None) or id(m))
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        all_matches.append(m)

        query_variants = expand_query(question, llm_client=None)
        attempted_variants = list(query_variants)
        _search_variants(query_variants, limit=retrieval_limit, score_threshold=threshold)

        # For comparison questions, also search each item individually.
        # The composite query "compare X and Y" often doesn't embed close enough
        # to individual product chunks, so we run a dedicated search per item
        # to guarantee context for both sides of the comparison.
        if is_comparison_question(question):
            comparison_items = extract_comparison_items(question)
            if comparison_items:
                _search_variants(
                    comparison_items,
                    limit=retrieval_limit,
                    score_threshold=threshold,
                )

        should_try_llm_rewrite = self.llm_client is not None and not is_list_question(question)
        if not all_matches and should_try_llm_rewrite:
            llm_variants = expand_query(
                question,
                llm_client=self.llm_client,
                allow_llm_rewrite=True,
            )
            retry_variants = [
                variant
                for variant in llm_variants
                if variant.lower() not in {existing.lower() for existing in attempted_variants}
            ]
            if retry_variants:
                attempted_variants.extend(retry_variants)
                _search_variants(retry_variants, limit=retrieval_limit, score_threshold=threshold)

        # Relaxed fallback
        if not all_matches:
            for variant in attempted_variants:
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
        lead_capture_enabled: bool = False,
        offering_scope: str | None = None,
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
            lead_capture_enabled=lead_capture_enabled,
            offering_scope=offering_scope,
        )
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            # Enhancement (prev session): raised cap from 0.1 → 0.7 so the LLM
            # can phrase answers naturally instead of always picking the single
            # most probable token.
            temperature=min(temperature, 0.7),
        )
        answer = (response.choices[0].message.content or FALLBACK_ANSWER).strip()
        career_followup = build_required_career_followup(
            question,
            answer,
            "\n\n".join(context_parts),
        )
        if career_followup:
            answer = f"{answer}\n\n{career_followup}".strip()
        followup = build_required_lead_capture_followup(
            question,
            answer,
            conversation_history,
            lead_capture_enabled=lead_capture_enabled,
        )
        if followup:
            answer = f"{answer}\n\n{followup}".strip()
        return answer

    # ── Enhancement 3: Streaming answer ───────────────────────────────────────

    def stream_answer_question(
        self,
        question: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        conversation_history: list[dict[str, str]] | None = None,
        prefetched_context: list[dict] | None = None,
        lead_capture_enabled: bool = False,
        offering_scope: str | None = None,
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
            lead_capture_enabled=lead_capture_enabled,
            offering_scope=offering_scope,
        )

        stream = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=min(temperature, 0.7),
            stream=True,
        )

        streamed_parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                streamed_parts.append(delta.content)
                yield delta.content

        full_answer = "".join(streamed_parts).strip()

        career_followup = build_required_career_followup(
            question,
            full_answer,
            "\n\n".join(context_parts),
        )
        if career_followup:
            yield "\n\n" + career_followup
            full_answer = f"{full_answer}\n\n{career_followup}".strip()

        followup = build_required_lead_capture_followup(
            question,
            full_answer,
            conversation_history,
            lead_capture_enabled=lead_capture_enabled,
        )
        if followup:
            yield "\n\n" + followup

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_context_parts(
        self, prefetched_context: list[dict] | None
    ) -> tuple[list[str], list[str]]:
        """Deduplicate and format context chunks; return (context_parts, source_urls)."""
        context_parts: list[str] = []
        source_urls: list[str] = []
        seen_urls: set[str] = set()
        seen_chunks: set[str] = set()
        total_chars = 0
        catalog_entries: list[str] = []

        for item in (prefetched_context or []):
            content = item.get("content", "").strip()
            if re.match(r"(?im)^\s*(?:Product|Service) Category\s*:", content):
                if content not in seen_chunks:
                    seen_chunks.add(content)
                    catalog_entries.append(content)
            source_name = item.get("source_name", "unknown")
            if source_name.startswith("http") and source_name not in seen_urls:
                seen_urls.add(source_name)
                source_urls.append(source_name)

        if catalog_entries:
            catalog_block = "[Source: catalog]\n" + "\n\n".join(catalog_entries)
            context_parts.append(catalog_block)
            total_chars = len(catalog_block)

        for item in (prefetched_context or []):
            content     = item.get("content", "").strip()
            source_name = item.get("source_name", "unknown")
            if re.match(r"(?im)^\s*(?:Product|Service) Category\s*:", content):
                continue
            if content and content not in seen_chunks:
                next_part = f"[Source: {source_name}]\n{content}"
                if context_parts:
                    projected_chars = total_chars + 2 + len(next_part)
                else:
                    projected_chars = total_chars + len(next_part)
                # Product:/Service: summary lines are compact (~50-250 chars each).
                # Skip the chunk-count limit for them so all catalog items always
                # reach the LLM regardless of how many product/service entries exist.
                # Only the character budget applies to these entries.
                is_catalog_summary = bool(
                    re.match(r"(?im)^\s*(?:Product|Service)\s*:", content)
                )
                over_chunk_limit = (
                    not is_catalog_summary
                    and len(context_parts) >= self.max_context_chunks
                )
                if over_chunk_limit or (context_parts and projected_chars > self.max_context_chars):
                    if over_chunk_limit:
                        break
                    continue
                seen_chunks.add(content)
                context_parts.append(next_part)
                total_chars = projected_chars
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