from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.rag_pipeline import RAGPipeline
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

_INTENT_RESPONSES: dict[str, str] = {
    "greeting":             "Hi! How can I help you today?",
    "gratitude":            "You're welcome! Let me know if you need anything else.",
    "acknowledgement":      "Great! Let me know if you need anything else.",
    "conversation_restart": "Welcome back! How can I help you?",
    "uncertain":            "No problem. Tell me what you're trying to do and I'll help.",
    "closing":              "Glad I could help. Have a great day!",
    "help_request":         "Of course! What do you need help with?",
    "followup_affirmation": "I'm not sure I understood that — could you clarify what you'd like to know more about? I want to make sure I give you the right information! 😊",
    "followup_negation":    "No problem! Let me know if there's anything else I can help you with. 😊",
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
    return _INTENT_RESPONSES.get(intent, "How can I help you?")


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


async def _build_context(
    pipeline: RAGPipeline,
    question: str,
    cs: ChunkStore,
    loop: asyncio.AbstractEventLoop,
) -> list[dict]:
    new_style_ids, legacy_chunks = await loop.run_in_executor(
        None, lambda: pipeline.search_chunks(question)
    )
    context: list[dict] = list(legacy_chunks)
    if new_style_ids:
        mongo_chunks = await cs.get_chunks_by_ids(new_style_ids)
        for c in mongo_chunks:
            context.append({
                "content": c.get("content", ""),
                "source_name": c.get("source_name", "unknown"),
            })
    return context


async def route_to_llm(
    question: str,
    pipeline: RAGPipeline,
    settings: dict,
    loop: asyncio.AbstractEventLoop,
    conversation_history: list[dict] | None = None,
    chunk_store: ChunkStore | None = None,
) -> str:
    if chunk_store is not None:
        context = await _build_context(pipeline, question, chunk_store, loop)
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


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    conversation_id: str | None = None


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
    # ── Billing status check (BEFORE anything else) ───────────────────────
    billing_status = await store.get_billing_status(user.id)
    if billing_status == "paused":
        raise HTTPException(status_code=402, detail="Your subscription is paused. Please contact support to resume.")
    if billing_status == "stopped":
        raise HTTPException(status_code=402, detail="Your subscription has been stopped. Please renew your plan to continue.")

    # ── Resolve agent (fetch real tenant_id from agent doc) ──────────────
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

    last_bot_message = ""
    if request.conversation_id:
        try:
            conv = await store.get_conversation(
                request.conversation_id, request.agent_id, tenant_id
            )
            last_bot_message = _get_last_bot_message(conv.get("messages", []))
        except Exception:
            pass

    intent = detect_intent(question, last_bot_message)
    if intent:
        await asyncio.sleep(0.6)
        answer = handle_intent(intent)
    else:
        pipeline = _build_pipeline(store, request.agent_id)
        try:
            loop = asyncio.get_event_loop()
            history = [{"role": "assistant", "content": last_bot_message}] if last_bot_message else None
            answer = await route_to_llm(question, pipeline, settings, loop,
                                        conversation_history=history, chunk_store=cs)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # ── Token consumption ─────────────────────────────────────────────────
    input_tokens   = max(len(question) // 4, 1)
    output_tokens  = max(len(answer)   // 4, 1)
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

    # ── Log fallback ──────────────────────────────────────────────────────
    if answer.strip().startswith("I don't have") or answer.strip().startswith("Hmm, I don't"):
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

    # ── Check agent owner's billing status ───────────────────────────────
    owner_user_id = agent_doc.get("user_id", "")
    if owner_user_id:
        billing_status = await store.get_billing_status(owner_user_id)
        if billing_status == "paused":
            raise HTTPException(status_code=402, detail="This service is currently paused.")
        if billing_status == "stopped":
            raise HTTPException(status_code=402, detail="This service is currently unavailable.")

    tenant_id = agent_doc["tenant_id"]
    question  = request.question.strip()

    last_bot_message = ""
    if request.conversation_id:
        try:
            conv = await store.get_conversation(
                request.conversation_id, request.agent_id, tenant_id
            )
            last_bot_message = _get_last_bot_message(conv.get("messages", []))
        except Exception:
            pass

    intent = detect_intent(question, last_bot_message)
    if intent:
        await asyncio.sleep(0.6)
        answer = handle_intent(intent)
    else:
        settings = await store.get_settings(request.agent_id, tenant_id)
        pipeline = _build_pipeline(store, request.agent_id)
        try:
            loop = asyncio.get_event_loop()
            history = [{"role": "assistant", "content": last_bot_message}] if last_bot_message else None
            answer = await route_to_llm(question, pipeline, settings, loop,
                                        conversation_history=history, chunk_store=cs)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if answer.strip().startswith("I don't have") or answer.strip().startswith("Hmm, I don't"):
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
