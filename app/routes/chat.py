from __future__ import annotations

import asyncio
import random
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.rag_pipeline import RAGPipeline, is_fallback_response
from app.prompt_builder import is_list_question
from app.services.admin_store_mongo import AdminStoreMongo
from app.services.chunk_store import ChunkStore

router = APIRouter(tags=["chat"])


# ── Intent Detection ──────────────────────────────────────────────────────────

_INTENT_PATTERNS: dict[str, list[str]] = {
    "greeting": [
        "hi", "hello", "hey", "hi there", "hey there", "hello there",
        "hey bot", "hello bot", "hi bot", "hello?", "hi!", "hey!",
        "anyone there", "anyone home", "good morning", "good afternoon",
        "good evening", "greetings", "howdy", "what's up", "whats up",
        "sup", "yo",
    ],
    "gratitude": [
        "thanks", "thank you", "thank you so much", "thanks a lot",
        "thanks a bunch", "thanks again", "many thanks", "much appreciated",
        "appreciate it", "appreciate that", "thx", "ty", "cheers",
        "grateful", "thanks for your help", "thank you for your help",
    ],
    "acknowledgement": [
        "ok", "okay", "ok got it", "okay got it", "got it", "i see",
        "understood", "makes sense", "cool", "nice", "great", "awesome",
        "perfect", "alright", "alrighty", "noted", "sounds good",
        "that makes sense",
    ],
    "conversation_restart": [
        "hello again", "hi again", "hey again", "are you there",
        "still there", "you there", "back again", "i'm back", "im back",
        "are you awake", "wake up",
    ],
    "uncertain": [
        "hmm", "hm", "umm", "um", "uh", "idk", "i don't know",
        "not sure", "maybe", "perhaps", "not really", "kind of",
        "sort of", "i guess",
    ],
    "closing": [
        "that's all", "thats all", "that's it", "thats it",
        "i'm done", "im done", "that helped", "that was helpful",
        "bye", "goodbye", "see you", "see ya", "take care",
        "have a good day", "have a great day", "thanks that's all",
        "thanks thats all", "no more questions", "nothing else",
        "all good", "all done",
    ],
    "help_request": [
        "help", "please help", "can you help", "can you help me",
        "i need help", "i need assistance", "assist me",
        "help me", "help me please", "i need support",
    ],
    # Enhancement 2: followup_affirmation NO LONGER has a static response.
    # These are routed to the LLM with last-answer context instead.
    "followup_affirmation": [
        "yes", "yes please", "yes!", "yep", "yeah", "yup",
        "please", "go ahead", "go on", "tell me more", "more", "more details",
        "more info", "more information", "continue", "and", "what else",
        "anything else", "more please", "yes tell me more",
    ],
    "followup_negation": [
        "no", "no thanks", "nope", "nah", "not really", "no need",
        "i'm good", "im good", "i am good", "no thank you", "that's fine",
        "that's enough", "thats enough", "i got it", "i get it",
    ],
}

# Enhancement 4: multiple response variants per intent — rotated randomly so
# users never hear the same phrase twice in a row.
_INTENT_RESPONSES: dict[str, list[str]] = {
    "greeting": [
        "Hi! How can I help you today?",
        "Hey there! What can I do for you?",
        "Hello! What would you like to know?",
        "Hi! Great to have you here — what's on your mind?",
    ],
    "gratitude": [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help! Anything else I can do for you?",
        "Glad that was useful! Feel free to ask if something else comes up.",
        "Of course! Don't hesitate to reach out if you have more questions.",
    ],
    "acknowledgement": [
        "Great! Let me know if you need anything else.",
        "Sounds good! Anything else you'd like to explore?",
        "Perfect! I'm here if you have more questions.",
        "Got it! Feel free to ask anything else.",
    ],
    "conversation_restart": [
        "Welcome back! How can I help you?",
        "Hey, good to see you again! What can I do for you?",
        "I'm here! What would you like to know?",
    ],
    "uncertain": [
        "No problem. Tell me what you're trying to do and I'll help.",
        "Take your time! What are you looking for?",
        "No worries — just let me know what you need and I'll do my best.",
    ],
    "closing": [
        "Glad I could help. Have a great day!",
        "It was a pleasure chatting with you! Take care.",
        "Thanks for stopping by! Hope to be helpful again soon.",
        "Glad to be of service! Have a wonderful day.",
    ],
    "help_request": [
        "Of course! What do you need help with?",
        "Sure, I'm here to help! What's the question?",
        "Absolutely! Tell me what you're looking for.",
    ],
    # Enhancement 2: followup_affirmation is handled by route_to_llm, not this dict.
    # This entry is kept as a safety fallback only.
    "followup_affirmation": [
        "Sure! Could you let me know which part you'd like to explore further?",
        "Happy to go deeper! Which aspect are you most interested in?",
    ],
    "followup_negation": [
        "No problem! Let me know if there's anything else I can help you with.",
        "All good! Just ask if something else comes to mind.",
        "Sure thing! I'm here whenever you need me.",
    ],
}

_YES_NO_QUESTION_ENDINGS = [
    "would you like", "do you want", "shall i", "should i", "can i", "may i",
    "are you interested", "is that correct", "does that help", "did you want",
    "have you tried", "will you", "sound good?", "make sense?",
    "want me to", "like me to", "want to know more about", "like to know more about",
]

_CLOSING_INVITATION_ENDINGS = [
    "let me know if you need more details", "let me know if you have any questions",
    "let me know if you need anything else", "let me know if there's anything else",
    "feel free to ask", "feel free to reach out", "hope that helps", "hope this helps",
    "anything else i can help", "anything else you'd like", "anything else you would like",
    "more details on any", "specific questions", "further questions", "further information",
    "more about any", "if you need more", "if you have more",
]


def bot_asked_yes_no_question(last_bot_message: str) -> bool:
    if not last_bot_message:
        return False
    lower = last_bot_message.lower().strip()
    return any(e in lower for e in _YES_NO_QUESTION_ENDINGS)


def detect_intent(message: str, last_bot_message: str = "") -> str | None:
    normalized = re.sub(r"[!?.]+$", "", message.strip().lower()).strip()

    if bot_asked_yes_no_question(last_bot_message):
        for intent, patterns in _INTENT_PATTERNS.items():
            if intent in ("followup_affirmation", "followup_negation"):
                continue
            for p in patterns:
                if normalized == p or normalized.startswith(p + " ") or normalized.startswith(p + ","):
                    return intent
        return None

    for intent, patterns in _INTENT_PATTERNS.items():
        for p in patterns:
            if normalized == p or normalized.startswith(p + " ") or normalized.startswith(p + ","):
                return intent
    return None


def handle_intent(intent: str) -> str:
    """Enhancement 4: pick a random variant so responses never sound canned."""
    variants = _INTENT_RESPONSES.get(intent)
    if variants:
        return random.choice(variants)
    return "How can I help you?"



# ── Reference & ordinal resolution ───────────────────────────────────────────
# Detects "tell me more about the first one / second one / that product" etc.
# and rewrites the query to the actual item name extracted from the last bot
# message BEFORE vector search — so the embeddings actually hit the right chunk.

_ORDINAL_MAP = {
    "first": 1,   "1st": 1,
    "second": 2,  "2nd": 2,
    "third": 3,   "3rd": 3,
    "fourth": 4,  "4th": 4,
    "fifth": 5,   "5th": 5,
    "sixth": 6,   "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8,  "8th": 8,
    "ninth": 9,   "9th": 9,
    "tenth": 10,  "10th": 10,
}

_REFERENCE_TRIGGERS = [
    "tell me more about", "more about", "what about", "explain",
    "details on", "detail about", "describe", "elaborate on",
    "give me more", "what is", "tell me about",
    "the first", "the second", "the third", "the fourth",
    "the fifth", "the sixth", "the seventh",
    # numeric ordinal suffixes
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    # bare digits as triggers (belt-and-suspenders alongside fullmatch check)
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
]


def _extract_numbered_items(text: str) -> dict[int, str]:
    """
    Parse a numbered list from a bot reply.
    Handles formats:  1. Item Name   or   1) Item Name
    Returns {1: "Item Name", 2: "Item Name", ...}
    """
    items: dict[int, str] = {}
    for m in re.finditer(r"(\d+)[.)\]\s+([^\n0-9][^\n]{2,})", text):
        idx = int(m.group(1))
        label = m.group(2).strip().rstrip(".,;:")
        if label:
            items[idx] = label
    return items


def resolve_reference_query(question: str, last_bot_message: str) -> str:
    """
    Rewrites ordinal/reference questions to the actual item name so the
    vector search retrieves the correct chunks.

    Handles:
      - "first one", "second one", "third" etc.  (word ordinals)
      - "the 4th", "2nd product" etc.            (numeric ordinals)
      - "tell me more about 3", "more on 7"      (triggered number)
      - "1", "2", "3" alone                      (bare number when list present)

    Always returns the original question on any error — never crashes the request.
    """
    if not last_bot_message:
        import logging; logging.getLogger("chatbot").warning(f"[RESOLVE] EMPTY last_bot_message for question={repr(question)}")
        return question
    try:
        q_lower = question.lower().strip()
        numbered = _extract_numbered_items(last_bot_message)
        import logging; logging.getLogger("chatbot").info(f"[RESOLVE] q={repr(question)} lbm_len={len(last_bot_message)} items={list(numbered.keys())}")

        # Nothing to resolve if the bot's last reply had no numbered list
        if not numbered:
            return question

        # ── Bare number only: "1", "2", "7" ──────────────────────────────
        # User typed just a digit — resolve directly if it maps to a list item
        bare = re.fullmatch(r"(\d+)", q_lower)
        if bare:
            idx = int(bare.group(1))
            if idx in numbered:
                return f"Tell me more about {numbered[idx]}"

        # ── Triggered number: "tell me about 3", "more on 2", "explain 4th" ──
        num_match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", q_lower)
        if num_match and any(t in q_lower for t in _REFERENCE_TRIGGERS):
            idx = int(num_match.group(1))
            if idx in numbered:
                return f"Tell me more about {numbered[idx]}"

        # ── Word ordinals: "first one", "the second", "third product" ────
        for word, idx in _ORDINAL_MAP.items():
            if re.search(rf"\b{re.escape(word)}\b", q_lower):
                if idx in numbered:
                    return f"Tell me more about {numbered[idx]}"

    except Exception:
        pass  # Never crash — fall through to original question

    return question

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


def _get_chunk_store() -> ChunkStore:
    from app.main import chunk_store
    return chunk_store


def _build_pipeline(store: AdminStoreMongo, agent_id: str) -> RAGPipeline:
    from app.main import agent_collection_name
    return RAGPipeline(
        pdf_directory=store.get_agent_pdf_dir(agent_id),
        website_directory=store.get_agent_website_dir(agent_id),
        snippets_directory=store.get_agent_snippet_dir(agent_id),
        qa_directory=store.get_agent_qa_dir(agent_id),
        collection_name=agent_collection_name(agent_id),
    )


def _get_last_bot_message(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


# Enhancement 1: load full conversation history (user + assistant pairs)
# instead of just the last bot message.
def _get_conversation_history(messages: list, max_turns: int = 3) -> list[dict]:
    """
    Return the last `max_turns` complete turns (user + assistant pairs) as a
    flat list of {"role": ..., "content": ...} dicts, oldest first.

    This lets the LLM resolve follow-up references like "tell me more about #3"
    or "which one is cheaper?" against the actual prior exchange.
    """
    # Walk backwards collecting pairs until we have max_turns
    collected: list[dict] = []
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            collected.insert(0, {"role": role, "content": content})
        if len(collected) >= max_turns * 2:
            break
    return collected


async def _build_context(
    pipeline: RAGPipeline,
    question: str,
    cs: ChunkStore,
    loop: asyncio.AbstractEventLoop,
    agent_id: str = "",
) -> list[dict]:
    """
    Build context chunks for the LLM.

    Summary-chunk strategy (MongoDB path)
    ──────────────────────────────────────
    For list questions ("what products do you offer?", "show me all solutions")
    we fetch every summary chunk for this agent DIRECTLY from MongoDB using a
    simple field filter (chunk_type="summary", agent_id=agent_id).

    This is more reliable than the Qdrant-scroll approach because:
    - MongoDB is the single source of truth for chunk content
    - No dependency on Qdrant payload filters being in sync
    - Guaranteed to return ALL products regardless of vector scores

    Summary chunks are prepended so they always appear first in the context
    window, giving the LLM a complete product list to work from.
    Detail chunks from vector search fill the remaining slots with richer info.
    """
    # ── Step 1: fetch summary chunks from MongoDB for list questions ──────────
    summary_context: list[dict] = []
    if agent_id and is_list_question(question):
        summary_ids = await cs.get_summary_chunk_ids_by_agent(agent_id)
        if summary_ids:
            summary_chunks = await cs.get_chunks_by_ids(summary_ids)
            for c in summary_chunks:
                content = c.get("content", "").strip()
                if content:
                    summary_context.append({
                        "content": content,
                        "source_name": c.get("source_name", "unknown"),
                    })

    # ── Step 2: normal vector search for detail chunks ────────────────────────
    new_style_ids, legacy_chunks = await loop.run_in_executor(
        None, lambda: pipeline.search_chunks(question)
    )
    detail_context: list[dict] = list(legacy_chunks)
    seen_content: set[str] = {c["content"] for c in summary_context}
    if new_style_ids:
        mongo_chunks = await cs.get_chunks_by_ids(new_style_ids)
        for c in mongo_chunks:
            content = c.get("content", "").strip()
            # Skip detail chunks that duplicate a summary chunk already added
            if content and content not in seen_content:
                detail_context.append({
                    "content": content,
                    "source_name": c.get("source_name", "unknown"),
                })

    # Summary chunks first, then detail chunks
    return summary_context + detail_context


async def route_to_llm(
    question: str,
    pipeline: RAGPipeline,
    settings: dict,
    loop: asyncio.AbstractEventLoop,
    conversation_history: list[dict] | None = None,
    chunk_store: ChunkStore | None = None,
    agent_id: str = "",
) -> str:
    if chunk_store is not None:
        context = await _build_context(pipeline, question, chunk_store, loop, agent_id=agent_id)
    else:
        context = []
    _ctx = context
    return await loop.run_in_executor(
        None,
        lambda: pipeline.answer_question(
            question,
            system_prompt=settings.get("system_prompt"),
            temperature=settings.get("temperature", 0.2),
            conversation_history=conversation_history,
            prefetched_context=_ctx,
        ),
    )


# ── Enhancement 7: accurate token counting with tiktoken ──────────────────────

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.encoding_for_model("gpt-4o-mini")

    def _count_tokens(text: str) -> int:
        return max(len(_TIKTOKEN_ENC.encode(text)), 1)

except Exception:
    # tiktoken not installed — fall back to the character heuristic
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(len(text) // 4, 1)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    conversation_id: str | None = None
    # Optional: frontend can pass the last bot message directly so reference
    # resolution ("1", "first one") works even without a conversation_id round-trip.
    last_bot_message: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str


# ── Protected chat ────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    # ── Billing status check ──────────────────────────────────────────────
    billing_status = await store.get_billing_status(user.id)
    if billing_status == "paused":
        raise HTTPException(status_code=402, detail="Your subscription is paused. Please contact support to resume.")
    if billing_status == "stopped":
        raise HTTPException(status_code=402, detail="Your subscription has been stopped. Please renew your plan to continue.")

    # ── Resolve agent ─────────────────────────────────────────────────────
    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent_doc = await store.get_agent_by_id(request.agent_id)
        else:
            agent_doc = await store.require_accessible_agent(request.agent_id, user.tenant_id, user.id)
        tenant_id = agent_doc["tenant_id"]
        settings = await store.get_settings(request.agent_id, tenant_id)
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    question = request.question.strip()

    # Load full conversation history from MongoDB
    conversation_history: list[dict] = []
    last_bot_message = ""
    if request.conversation_id:
        try:
            conv = await store.get_conversation(
                request.conversation_id, request.agent_id, tenant_id
            )
            msgs = conv.get("messages", [])
            last_bot_message = _get_last_bot_message(msgs)
            conversation_history = _get_conversation_history(msgs, max_turns=3)
        except Exception:
            pass

        # Fallback 1: frontend-supplied last_bot_message (when conversation_id is null)
    if not last_bot_message and request.last_bot_message and request.last_bot_message.strip():
        last_bot_message = request.last_bot_message.strip()

    # Fallback 2: most recent saved conversation for this agent
    # Handles page-refresh / new-session where conversation_id resets to null
    if not last_bot_message:
        try:
            recent = await store.db.conversations.find_one(
                {"agent_id": request.agent_id, "tenant_id": tenant_id},
                sort=[("updated_at", -1)]
            )
            if recent:
                recent_msgs = recent.get("messages", [])
                last_bot_message = _get_last_bot_message(recent_msgs)
                if not conversation_history:
                    conversation_history = _get_conversation_history(recent_msgs, max_turns=3)
        except Exception:
            pass

    # Reference resolution: rewrite "1"/"first one" → actual product name
    # BEFORE intent detection and BEFORE vector search.
    question = resolve_reference_query(question, last_bot_message)

    intent = detect_intent(question, last_bot_message)

    # Enhancement 2: route followup_affirmation to the LLM with context
    # instead of returning a static confused message.
    if intent == "followup_affirmation" and last_bot_message:
        intent = None  # fall through to LLM with history already loaded

    if intent:
        await asyncio.sleep(0.6)
        answer = handle_intent(intent)
    else:
        pipeline = _build_pipeline(store, request.agent_id)
        try:
            loop = asyncio.get_event_loop()
            answer = await route_to_llm(
                question, pipeline, settings, loop,
                conversation_history=conversation_history,
                chunk_store=cs,
                agent_id=request.agent_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Enhancement 7: accurate token counting via tiktoken
    input_tokens   = _count_tokens(question)
    output_tokens  = _count_tokens(answer)
    summary_tokens = 150

    consumption = await store.consume_tokens(
        user_id=user.id,
        agent_id=request.agent_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        summary_tokens=summary_tokens,
    )
    if not consumption["allowed"] and consumption["reason"] != "no_plan":
        raise HTTPException(status_code=429, detail=consumption["reason"])

    # Enhancement 8: use is_fallback_response() from rag_pipeline for reliable detection
    if is_fallback_response(answer):
        try:
            await store.log_fallback(
                agent_id=request.agent_id,
                tenant_id=tenant_id,
                question=question,
                conversation_id=request.conversation_id,
            )
        except Exception:
            pass

    conversation = await store.append_conversation_messages(
        agent_id=request.agent_id,
        tenant_id=tenant_id,
        user_id=user.id,
        user_message=question,
        assistant_message=answer,
        conversation_id=request.conversation_id,
    )
    return ChatResponse(answer=answer, conversation_id=conversation["id"])


# ── Enhancement 3: Streaming protected chat ───────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    """
    Streaming version of /chat. Tokens are pushed as Server-Sent Events
    so the first words appear in ~300ms instead of waiting for the full response.

    Frontend usage:
        const es = new EventSource('/chat/stream', { method: 'POST', ... })
        OR use fetch() with ReadableStream and split on '\n\ndata: '
    """
    billing_status = await store.get_billing_status(user.id)
    if billing_status == "paused":
        raise HTTPException(status_code=402, detail="Your subscription is paused.")
    if billing_status == "stopped":
        raise HTTPException(status_code=402, detail="Your subscription has been stopped.")

    try:
        if user.role == UserRole.SUPER_ADMIN:
            agent_doc = await store.get_agent_by_id(request.agent_id)
        else:
            agent_doc = await store.require_accessible_agent(request.agent_id, user.tenant_id, user.id)
        tenant_id = agent_doc["tenant_id"]
        settings = await store.get_settings(request.agent_id, tenant_id)
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    question = request.question.strip()

    conversation_history: list[dict] = []
    last_bot_message = ""
    if request.conversation_id:
        try:
            conv = await store.get_conversation(request.conversation_id, request.agent_id, tenant_id)
            msgs = conv.get("messages", [])
            last_bot_message = _get_last_bot_message(msgs)
            conversation_history = _get_conversation_history(msgs, max_turns=3)
        except Exception:
            pass

    # Reference resolution for streaming endpoint
    question = resolve_reference_query(question, last_bot_message)

    intent = detect_intent(question, last_bot_message)
    if intent == "followup_affirmation" and last_bot_message:
        intent = None

    pipeline = _build_pipeline(store, request.agent_id)
    loop = asyncio.get_event_loop()

    if intent:
        # For intent responses just stream the single string immediately
        answer_text = handle_intent(intent)

        async def intent_stream():
            yield f"data: {answer_text}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(intent_stream(), media_type="text/event-stream")

    # Build context in async land before entering the sync generator
    context = await _build_context(pipeline, question, cs, loop, agent_id=request.agent_id)
    full_answer_parts: list[str] = []

    async def token_stream():
        try:
            gen = await loop.run_in_executor(
                None,
                lambda: pipeline.stream_answer_question(
                    question,
                    system_prompt=settings.get("system_prompt"),
                    temperature=settings.get("temperature", 0.2),
                    conversation_history=conversation_history,
                    prefetched_context=context,
                ),
            )
            for token in gen:
                full_answer_parts.append(token)
                # Escape newlines so SSE framing is not broken
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            # Persist conversation + token usage after streaming completes
            full_answer = "".join(full_answer_parts)
            if full_answer:
                try:
                    await store.consume_tokens(
                        user_id=user.id,
                        agent_id=request.agent_id,
                        input_tokens=_count_tokens(question),
                        output_tokens=_count_tokens(full_answer),
                        summary_tokens=150,
                    )
                    if is_fallback_response(full_answer):
                        await store.log_fallback(
                            agent_id=request.agent_id,
                            tenant_id=tenant_id,
                            question=question,
                            conversation_id=request.conversation_id,
                        )
                    await store.append_conversation_messages(
                        agent_id=request.agent_id,
                        tenant_id=tenant_id,
                        user_id=user.id,
                        user_message=question,
                        assistant_message=full_answer,
                        conversation_id=request.conversation_id,
                    )
                except Exception:
                    pass

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ── Widget chat (public) ──────────────────────────────────────────────────────

@router.post("/widget/chat", response_model=ChatResponse)
async def widget_chat(
    request: ChatRequest,
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
):
    agent_doc = await store.db.agents.find_one({"_id": request.agent_id})
    if not agent_doc:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_id}' not found.")

    owner_user_id = agent_doc.get("user_id", "")
    if owner_user_id:
        billing_status = await store.get_billing_status(owner_user_id)
        if billing_status == "paused":
            raise HTTPException(status_code=402, detail="This service is currently paused.")
        if billing_status == "stopped":
            raise HTTPException(status_code=402, detail="This service is currently unavailable.")

    tenant_id = agent_doc["tenant_id"]
    question  = request.question.strip()

    # Load full conversation history from MongoDB
    conversation_history: list[dict] = []
    last_bot_message = ""
    if request.conversation_id:
        try:
            conv = await store.get_conversation(
                request.conversation_id, request.agent_id, tenant_id
            )
            msgs = conv.get("messages", [])
            last_bot_message = _get_last_bot_message(msgs)
            conversation_history = _get_conversation_history(msgs, max_turns=3)
        except Exception:
            pass

    # Fallback 1: frontend-supplied value
    if not last_bot_message and request.last_bot_message and request.last_bot_message.strip():
        last_bot_message = request.last_bot_message.strip()

    # Fallback 2: most recent conversation for this agent
    if not last_bot_message:
        try:
            recent = await store.db.conversations.find_one(
                {"agent_id": request.agent_id, "tenant_id": tenant_id},
                sort=[("updated_at", -1)]
            )
            if recent:
                recent_msgs = recent.get("messages", [])
                last_bot_message = _get_last_bot_message(recent_msgs)
                if not conversation_history:
                    conversation_history = _get_conversation_history(recent_msgs, max_turns=3)
        except Exception:
            pass

    # Reference resolution
    question = resolve_reference_query(question, last_bot_message)

    intent = detect_intent(question, last_bot_message)

    # Enhancement 2: followup_affirmation → LLM
    if intent == "followup_affirmation" and last_bot_message:
        intent = None

    if intent:
        await asyncio.sleep(0.6)
        answer = handle_intent(intent)
    else:
        settings = await store.get_settings(request.agent_id, tenant_id)
        pipeline = _build_pipeline(store, request.agent_id)
        try:
            loop = asyncio.get_event_loop()
            answer = await route_to_llm(
                question, pipeline, settings, loop,
                conversation_history=conversation_history,
                chunk_store=cs,
                agent_id=request.agent_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Enhancement 8: reliable fallback detection
    if is_fallback_response(answer):
        try:
            await store.log_fallback(
                agent_id=request.agent_id,
                tenant_id=tenant_id,
                question=question,
                conversation_id=request.conversation_id,
            )
        except Exception:
            pass

    conversation = await store.append_conversation_messages(
        agent_id=request.agent_id,
        tenant_id=tenant_id,
        user_id="widget",
        user_message=question,
        assistant_message=answer,
        conversation_id=request.conversation_id,
    )
    return ChatResponse(answer=answer, conversation_id=conversation["id"])


# ── Widget settings (public) ──────────────────────────────────────────────────

class PublicSettingsResponse(BaseModel):
    display_name: str
    welcome_message: str
    primary_color: str
    appearance: str


@router.get("/widget/settings/{agent_id}", response_model=PublicSettingsResponse)
async def widget_settings(
    agent_id: str,
    store: AdminStoreMongo = Depends(_get_store),
):
    agent_doc = await store.db.agents.find_one({"_id": agent_id})
    if not agent_doc:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    s = agent_doc.get("settings", {})
    return PublicSettingsResponse(
        display_name=s.get("display_name") or agent_doc.get("name", "Assistant"),
        welcome_message=s.get("welcome_message") or "Hi! What can I help you with?",
        primary_color=s.get("primary_color") or "#0f172a",
        appearance=s.get("appearance") or "light",
    )