from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, get_current_user
from app.rag_pipeline import RAGPipeline
from app.services.admin_store_mongo import AdminStoreMongo

router = APIRouter(tags=["chat"])


def _get_store() -> AdminStoreMongo:
    from app.main import store
    return store


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
    store: AdminStoreMongo = Depends(_get_store),
):
    try:
        agent = await store.require_agent(request.agent_id, user.tenant_id)
        settings = await store.get_settings(request.agent_id, user.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from app.main import agent_collection_name
    pipeline = RAGPipeline(
        pdf_directory=store.get_agent_pdf_dir(request.agent_id),
        website_directory=store.get_agent_website_dir(request.agent_id),
        snippets_directory=store.get_agent_snippet_dir(request.agent_id),
        qa_directory=store.get_agent_qa_dir(request.agent_id),
        collection_name=agent_collection_name(request.agent_id),
    )

    try:
        answer = pipeline.answer_question(
            request.question.strip(),
            system_prompt=settings.get("system_prompt"),
            temperature=settings.get("temperature", 0.2),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    conversation = await store.append_conversation_messages(
        agent_id=request.agent_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
        user_message=request.question.strip(),
        assistant_message=answer,
        conversation_id=request.conversation_id,
    )

    return ChatResponse(answer=answer, conversation_id=conversation["id"])
