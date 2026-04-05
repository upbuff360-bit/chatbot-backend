from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import random
import re
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, get_current_user
from app.models.user import UserRole
from app.rag_pipeline import RAGPipeline, is_fallback_response
from app.prompt_builder import (
    build_required_lead_capture_followup,
    is_comparison_question,
    detect_pricing_subject,
    detect_list_category,
    has_recommendation_requirements,
    is_list_question,
    is_pricing_question,
    is_recommendation_question,
)
from app.services.admin_store_mongo import AdminStoreMongo
from app.services.chunk_store import ChunkStore

router = APIRouter(tags=["chat"])


def _normalize_website_url_for_match(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    return re.sub(r"(?<!:)/{2,}", "/", value)


def _sse_data(value: str) -> str:
    return f"data: {value}\n\n"


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _apply_required_lead_followup(
    answer: str,
    *,
    question: str,
    conversation_history: list[dict] | None,
    lead_capture_enabled: bool,
) -> str:
    followup = build_required_lead_capture_followup(
        question,
        answer,
        conversation_history,
        lead_capture_enabled=lead_capture_enabled,
    )
    if not followup:
        return answer
    if followup in answer:
        return answer
    return f"{answer.rstrip()}\n\n{followup}".strip()


# ── Intent Detection ──────────────────────────────────────────────────────────

_INTENT_PATTERNS: dict[str, list[str]] = {
    "identity": [
        "who are you", "who r you", "what are you", "what is your name",
        "what's your name", "whats your name", "your name", "tell me your name",
        "may i know your name", "can i know your name", "introduce yourself",
        "who am i talking to", "who am i chatting with", "tell me about yourself",
    ],
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

_DEFAULT_STARTER_SUGGESTIONS = (
    "What do you offer?",
    "What are your services?",
    "What are your products?",
    "Which solution would you recommend?",
)
_STARTER_LABEL_SKIP_RE = re.compile(
    r"(?i)^(?:\d+\s*(?:min|minute|minutes)\s+read|read more|view all|our clients|all industries|"
    r"product finder|service finder|downloads?|company history|history|privacy policy|terms(?:\s*&\s*conditions)?|"
    r"contact|customer service|location|locations?|current openings?|open positions?|job openings?|"
    r"career|careers|job|jobs|vacanc(?:y|ies)|hiring|apply now|join our team|join our email list|"
    r"discover|reach us|news room|resource library|view all resources|find distributors|location finder)$|"
    r"\b(?:min read|read more|blog|article|newsroom|case study|our efficient .* process|"
    r"manufacturing process|process|download|career|careers|job|jobs|vacanc(?:y|ies)|hiring|apply now)\b"
)
_STARTER_LABEL_SIZE_ONLY_RE = re.compile(
    r"(?i)^\d+(?:\.\d+)?\s*(?:l|ltr|litre|litres|ml|kg|kgs|g|gm|gram|grams|pack|packs|pcs?|piece|pieces)\b\.?$"
)
_GENERIC_AGENT_STARTER_SUGGESTIONS = (
    "What do you offer?",
    "How can you help?",
    "Tell me about your company",
    "How do I get started?",
)

_PRODUCT_TOPIC_RE = re.compile(
    r"\b(product|products|solution|solutions|platform|platforms|crm|"
    r"field service|warehouse|inventory|manufacturing|shopfloor|"
    r"asset tracking|partner portal|partner portals|configurator)\b",
    re.IGNORECASE,
)
_SERVICE_TOPIC_RE = re.compile(
    r"\b(service|services|consulting|implementation|integration|"
    r"migration|training|support|maintenance|onboarding|managed service|"
    r"professional service)\b",
    re.IGNORECASE,
)
_OFFERING_TOPIC_RE = re.compile(
    r"\b(offer|offers|offering|offerings|provide|provides|what do you do|"
    r"what can you do)\b",
    re.IGNORECASE,
)
_GENERAL_TOPIC_RE = re.compile(
    r"\b(about|company|business|who are you|how can you help|get started|"
    r"contact|overview|background)\b",
    re.IGNORECASE,
)

_GREETING_PREFIX_RE = re.compile(
    r"^\s*(?:(?:good morning|good afternoon|good evening|good night|"
    r"hello again|hi again|hey again|hi there|hey there|hello there|"
    r"hello|hi|hey|howdy|greetings|yo)\b[\s,!.?:-]*)+",
    re.IGNORECASE,
)
_TIME_GREETING_PHRASES = (
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
)
_GENERAL_GREETING_PHRASES = (
    "hello again",
    "hi again",
    "hey again",
    "hello there",
    "hi there",
    "hey there",
    "hello",
    "hi",
    "hey",
    "howdy",
    "greetings",
    "yo",
)


def bot_asked_yes_no_question(last_bot_message: str) -> bool:
    if not last_bot_message:
        return False
    lower = last_bot_message.lower().strip()
    return any(e in lower for e in _YES_NO_QUESTION_ENDINGS)


def _normalize_for_greeting_match(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _phrase_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _consume_fuzzy_greeting_phrase(
    tokens: list[str],
    start: int,
    phrases: tuple[str, ...],
    *,
    threshold: float,
) -> int:
    if start >= len(tokens):
        return 0

    best_len = 0
    best_score = 0.0
    for phrase in phrases:
        phrase_tokens = phrase.split()
        end = start + len(phrase_tokens)
        if end > len(tokens):
            continue
        candidate = " ".join(tokens[start:end])
        score = _phrase_similarity(candidate, phrase)
        if score >= threshold and score > best_score:
            best_score = score
            best_len = len(phrase_tokens)
    return best_len


def _greeting_prefix_token_count(message: str) -> int:
    normalized = _normalize_for_greeting_match(message)
    tokens = normalized.split()
    if not tokens:
        return 0

    consumed = 0
    consumed += _consume_fuzzy_greeting_phrase(
        tokens, consumed, _GENERAL_GREETING_PHRASES, threshold=0.84
    )
    consumed += _consume_fuzzy_greeting_phrase(
        tokens, consumed, _TIME_GREETING_PHRASES, threshold=0.78
    )

    if consumed == 0:
        # Allow direct time greeting with a typo, e.g. "goor afternoon".
        consumed = _consume_fuzzy_greeting_phrase(
            tokens, 0, _TIME_GREETING_PHRASES, threshold=0.78
        )

    return consumed


def _is_greeting_only_message(message: str) -> bool:
    normalized = _normalize_for_greeting_match(message)
    tokens = normalized.split()
    if not tokens:
        return False
    return _greeting_prefix_token_count(message) == len(tokens)


def _detect_time_greeting(message: str) -> str | None:
    normalized = _normalize_for_greeting_match(message)
    if not normalized:
        return None

    tokens = normalized.split()
    for start in range(min(2, len(tokens))):
        consumed = _consume_fuzzy_greeting_phrase(
            tokens, start, _TIME_GREETING_PHRASES, threshold=0.78
        )
        if not consumed:
            continue
        candidate = " ".join(tokens[start:start + consumed])
        best_phrase = max(
            _TIME_GREETING_PHRASES,
            key=lambda phrase: _phrase_similarity(candidate, phrase),
        )
        if "morning" in best_phrase:
            return "morning"
        if "afternoon" in best_phrase:
            return "afternoon"
        if "evening" in best_phrase:
            return "evening"
        if "night" in best_phrase:
            return "night"
    return None


def detect_intent(message: str, last_bot_message: str = "") -> str | None:
    normalized = re.sub(r"[!?.]+$", "", message.strip().lower()).strip()
    stripped_after_greeting = _strip_greeting_prefix(message)
    has_non_greeting_content = bool(
        stripped_after_greeting and stripped_after_greeting != message.strip()
    )

    if _is_greeting_only_message(message):
        return "greeting"

    if bot_asked_yes_no_question(last_bot_message):
        for intent, patterns in _INTENT_PATTERNS.items():
            if has_non_greeting_content and intent == "greeting":
                continue
            if intent in ("followup_affirmation", "followup_negation"):
                continue
            for p in patterns:
                if normalized == p or normalized.startswith(p + " ") or normalized.startswith(p + ","):
                    return intent
        return None

    for intent, patterns in _INTENT_PATTERNS.items():
        if has_non_greeting_content and intent == "greeting":
            continue
        for p in patterns:
            if normalized == p or normalized.startswith(p + " ") or normalized.startswith(p + ","):
                return intent
    return None


def _matching_greeting_response(message: str) -> str | None:
    time_greeting = _detect_time_greeting(message)

    if time_greeting == "morning":
        return random.choice([
            "Good morning! How can I help you today?",
            "Good morning! What can I do for you?",
            "Good morning! What would you like to know?",
        ])
    if time_greeting == "afternoon":
        return random.choice([
            "Good afternoon! How can I help you today?",
            "Good afternoon! What can I do for you?",
            "Good afternoon! What would you like to know?",
        ])
    if time_greeting == "evening":
        return random.choice([
            "Good evening! How can I help you today?",
            "Good evening! What can I do for you?",
            "Good evening! What would you like to know?",
        ])
    if time_greeting == "night":
        return random.choice([
            "Good night! I'm here if you need anything before you head off.",
            "Good night! Let me know if there's anything you'd like help with.",
        ])

    return None


def _resolve_agent_name(agent_doc: dict | None) -> str:
    if not agent_doc:
        return "our"

    for key in ("name", "display_name", "website_name"):
        value = str(agent_doc.get(key) or "").strip()
        if value:
            return value
    return "our"


def _identity_response(agent_doc: dict | None) -> str:
    agent_name = _resolve_agent_name(agent_doc)
    if agent_name.lower().endswith("ai agent"):
        full_name = agent_name
    else:
        full_name = f"{agent_name} AI agent"

    return (
        f"I'm {full_name}. I'm here to assist you with any questions related to "
        f"{agent_name}, and I'll be happy to help in any way I can. "
        "Please let me know how I may assist you."
    )


def _dedupe_suggestions(items: list[str], *, limit: int = 4) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        value = str(item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
        if len(cleaned) >= limit:
            break
    return cleaned


def _starter_suggestions(agent_doc: dict | None = None) -> list[str]:
    return _dedupe_suggestions(list(_DEFAULT_STARTER_SUGGESTIONS))


def _clean_starter_label(label: str | None) -> str | None:
    value = " ".join(str(label or "").split()).strip(" -|")
    if not value:
        return None

    pipe_parts = [part.strip() for part in re.split(r"\s+\|\s+", value) if part.strip()]
    if pipe_parts:
        preferred = pipe_parts[0]
        if preferred and len(preferred) <= 80:
            value = preferred

    value = value.strip(" -|")
    if not value or len(value) > 90:
        return None
    if _STARTER_LABEL_SIZE_ONLY_RE.fullmatch(value):
        return None
    if _STARTER_LABEL_SKIP_RE.search(value):
        return None
    return value


async def _starter_suggestions_for_agent(
    agent_doc: dict | None,
    cs: ChunkStore | None = None,
) -> list[str]:
    if not agent_doc or cs is None:
        return _starter_suggestions(agent_doc)

    agent_id = str(agent_doc.get("_id") or "").strip()
    if not agent_id:
        return _starter_suggestions(agent_doc)

    async def _load_summary_chunks(category: str) -> list[dict]:
        summary_ids = await cs.get_summary_chunk_ids_by_agent(agent_id, category=category)
        if not summary_ids:
            return []
        return await cs.get_chunks_by_ids(summary_ids)

    product_chunks, service_chunks = await asyncio.gather(
        _load_summary_chunks("product"),
        _load_summary_chunks("service"),
    )

    def _clean_chunks(chunks: list[dict], *, category: str) -> list[dict]:
        cleaned: list[dict] = []
        seen_labels: set[str] = set()
        for chunk in chunks:
            content = str(chunk.get("content") or "").strip()
            if not content or _should_skip_summary_for_list(content, category):
                continue
            label = _clean_starter_label(_extract_summary_label(content))
            if not label:
                continue
            normalized = _normalize_summary_label(label)
            if normalized in seen_labels:
                continue
            seen_labels.add(normalized)
            cleaned.append(chunk)
        return cleaned

    product_chunks = _clean_chunks(product_chunks, category="product")
    service_chunks = _clean_chunks(service_chunks, category="service")

    product_website_categories = [
        {
            "label": clean_label,
            "count": 0,
            "items": [],
        }
        for label in _get_agent_website_catalog_categories(agent_id, list_category="product")
        if (clean_label := _clean_starter_label(label))
    ]
    service_website_categories = [
        {
            "label": clean_label,
            "count": 0,
            "items": [],
        }
        for label in _get_agent_website_catalog_categories(agent_id, list_category="service")
        if (clean_label := _clean_starter_label(label))
    ]

    product_categories = (
        product_website_categories
        if len(product_website_categories) >= 3
        else _build_catalog_groups(product_chunks, list_category="product")
    )
    if len(product_categories) < 3 and product_website_categories:
        product_categories = product_website_categories

    service_categories = (
        service_website_categories
        if len(service_website_categories) >= 3
        else _build_catalog_groups(service_chunks, list_category="service")
    )
    if len(service_categories) < 3 and service_website_categories:
        service_categories = service_website_categories

    product_labels = [_clean_starter_label(_extract_summary_label(chunk.get("content", ""))) for chunk in product_chunks]
    product_labels = [label for label in product_labels if label]
    service_labels = [_clean_starter_label(_extract_summary_label(chunk.get("content", ""))) for chunk in service_chunks]
    service_labels = [label for label in service_labels if label]

    if product_categories:
        product_labels = []
    if service_categories:
        service_labels = []

    has_product_starters = bool(product_categories or product_labels)
    has_service_starters = bool(service_categories or len(service_labels) >= 2)

    if has_product_starters and has_service_starters:
        suggestions: list[str] = [
            "What products do you offer?",
            "What services do you offer?",
        ]
        if product_categories:
            suggestions.append(f"Tell me about {product_categories[0]['label']}")
        elif product_labels:
            suggestions.append(f"Tell me about {product_labels[0]}")
        if service_categories:
            suggestions.append(f"Tell me about {service_categories[0]['label']}")
        elif service_labels:
            suggestions.append(f"Tell me about {service_labels[0]}")
        return _dedupe_suggestions(suggestions)

    if has_product_starters:
        if len(product_categories) >= 3:
            suggestions = [
                "What product categories do you offer?",
                *(f"Tell me about {group['label']}" for group in product_categories[:3]),
            ]
            return _dedupe_suggestions(suggestions)

        suggestions = ["What products do you offer?"]
        suggestions.extend(f"Tell me about {label}" for label in product_labels[:3])
        if len(suggestions) < 4:
            suggestions.append("Which product would you recommend?")
        return _dedupe_suggestions(suggestions)

    if has_service_starters:
        if len(service_categories) >= 3:
            suggestions = [
                "What service categories do you offer?",
                *(f"Tell me about {group['label']}" for group in service_categories[:3]),
            ]
            return _dedupe_suggestions(suggestions)

        suggestions = ["What services do you offer?"]
        suggestions.extend(f"Tell me about {label}" for label in service_labels[:3])
        if len(suggestions) < 4:
            suggestions.append("Which service would you recommend?")
        return _dedupe_suggestions(suggestions)

    return _dedupe_suggestions(list(_GENERIC_AGENT_STARTER_SUGGESTIONS))


def _detect_topic_from_text(text: str) -> tuple[str | None, str | None]:
    lower = (text or "").lower().strip()
    if not lower:
        return None, None

    if is_pricing_question(lower):
        return "pricing", detect_pricing_subject(lower)

    list_category = detect_list_category(lower) if is_list_question(lower) else None
    if list_category:
        return list_category, None

    if _PRODUCT_TOPIC_RE.search(lower):
        return "product", None
    if _SERVICE_TOPIC_RE.search(lower):
        return "service", None
    if _OFFERING_TOPIC_RE.search(lower):
        return "offering", None
    if _GENERAL_TOPIC_RE.search(lower):
        return "general", None

    return None, None


def _last_non_pricing_topic(conversation_history: list[dict] | None) -> str | None:
    if not conversation_history:
        return None

    for msg in reversed(conversation_history):
        category, _ = _detect_topic_from_text(str(msg.get("content") or ""))
        if category and category != "pricing":
            return category
    return None


def _infer_suggestion_topic(
    question: str,
    answer: str,
    conversation_history: list[dict] | None = None,
) -> tuple[str, str | None]:
    category, pricing_subject = _detect_topic_from_text(question)
    if category == "pricing":
        if pricing_subject:
            return category, pricing_subject
        history_topic = _last_non_pricing_topic(conversation_history)
        if history_topic == "product":
            return category, "products"
        if history_topic == "service":
            return category, "services"
        if history_topic == "offering":
            return category, "products and services"
        answer_category, _ = _detect_topic_from_text(answer)
        if answer_category == "product":
            return category, "products"
        if answer_category == "service":
            return category, "services"
        if answer_category == "offering":
            return category, "products and services"
        return category, None

    if category:
        return category, pricing_subject

    answer_category, _ = _detect_topic_from_text(answer)
    if answer_category and answer_category != "pricing":
        return answer_category, None

    history_topic = _last_non_pricing_topic(conversation_history)
    if history_topic:
        return history_topic, None

    if answer_category == "pricing":
        return "pricing", None

    return "general", None


def _clean_followup_label(label: str | None) -> str | None:
    value = " ".join(str(label or "").split()).strip(" -|.,;:")
    if not value:
        return None
    if len(value) > 110:
        return None
    if _STARTER_LABEL_SIZE_ONLY_RE.fullmatch(value):
        return None
    if _STARTER_LABEL_SKIP_RE.search(value):
        return None
    return value


def _extract_grounded_answer_items(answer: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()

    for label in _extract_numbered_items(answer).values():
        cleaned = _clean_followup_label(label)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(cleaned)
        if len(items) >= 4:
            return items

    for line in str(answer or "").splitlines():
        match = re.match(r"(?i)^\s*(?:Product|Service)\s+Category\s*:\s*(.+?)\s*$", line.strip())
        if not match:
            continue
        cleaned = _clean_followup_label(match.group(1))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(cleaned)
        if len(items) >= 4:
            break

    return items


def _same_category_comparison_suggestions_from_chunks(
    question: str,
    summary_chunks: list[dict],
    *,
    agent_id: str = "",
) -> list[str]:
    comparison_scope = _extract_same_category_comparison_scope(
        question,
        summary_chunks,
        agent_id=agent_id,
    )
    focus_entry = comparison_scope.get("focus_entry") or {}
    focus_label = str(focus_entry.get("label") or "").strip()
    focus_category = str(focus_entry.get("main_category") or "").strip()
    focus_chunk_category = str(focus_entry.get("chunk_category") or "").strip()
    if not focus_label or not focus_category or focus_chunk_category not in {"product", "service"}:
        return []

    mentioned_labels = {
        _normalize_summary_label(str(entry.get("label") or ""))
        for entry in (comparison_scope.get("matched_entries") or [])
        if str(entry.get("label") or "").strip()
    }

    peer_labels: list[str] = []
    seen_peer_labels: set[str] = set()
    for chunk in summary_chunks:
        if str(chunk.get("category") or "") != focus_chunk_category:
            continue
        content = str(chunk.get("content") or "").strip()
        chunk_main_category = _extract_summary_field(content, "category")
        if _normalize_summary_label(chunk_main_category) != _normalize_summary_label(focus_category):
            continue
        label = _clean_followup_label(_extract_summary_label(content))
        if not label:
            continue
        normalized_label = _normalize_summary_label(label)
        if normalized_label == _normalize_summary_label(focus_label):
            continue
        if normalized_label in seen_peer_labels:
            continue
        seen_peer_labels.add(normalized_label)
        peer_labels.append(label)

    suggestions: list[str] = []
    if comparison_scope.get("cross_category"):
        for label in peer_labels[:3]:
            suggestions.append(f"Compare {focus_label} with {label}")
        suggestions.append(f"Tell me more about {focus_label}")
        return _dedupe_suggestions(suggestions)

    for label in peer_labels:
        normalized_label = _normalize_summary_label(label)
        if normalized_label in mentioned_labels:
            continue
        suggestions.append(f"Compare {focus_label} with {label}")
        if len(suggestions) >= 3:
            break
    suggestions.append(f"Tell me more about {focus_label}")
    return _dedupe_suggestions(suggestions)


async def _build_comparison_suggestions_for_agent(
    question: str,
    agent_id: str,
    cs: ChunkStore | None,
) -> list[str]:
    if not cs or not agent_id or not is_comparison_question(question):
        return []

    summary_ids = await cs.get_summary_chunk_ids_by_agent(agent_id)
    if not summary_ids:
        return []

    chunks = await cs.get_chunks_by_ids(summary_ids)
    cleaned_chunks: list[dict] = []
    seen_labels: set[tuple[str, str]] = set()
    for chunk in chunks:
        chunk_category = str(chunk.get("category") or "")
        if chunk_category not in {"product", "service"}:
            continue
        content = str(chunk.get("content") or "").strip()
        label = _clean_followup_label(_extract_summary_label(content))
        if not content or not label:
            continue
        dedupe_key = (chunk_category, _normalize_summary_label(label))
        if dedupe_key in seen_labels:
            continue
        seen_labels.add(dedupe_key)
        cleaned_chunks.append({
            "content": content,
            "source_name": str(chunk.get("source_name") or "unknown"),
            "category": chunk_category,
        })

    return _same_category_comparison_suggestions_from_chunks(
        question,
        cleaned_chunks,
        agent_id=agent_id,
    )


def _topic_suggestions(
    category: str,
    *,
    question: str,
    answer: str,
    conversation_history: list[dict] | None = None,
    pricing_subject: str | None = None,
) -> list[str]:
    if category == "pricing":
        if pricing_subject == "products":
            return _dedupe_suggestions([
                "Do you have pricing details for your products?",
                "Can I get a product quote?",
                "Is product pricing customized?",
                "What affects product pricing?",
            ])
        if pricing_subject == "services":
            return _dedupe_suggestions([
                "Do you have pricing details for your services?",
                "Can I get a service quote?",
                "Is service pricing customized?",
                "What affects service pricing?",
            ])
        return _dedupe_suggestions([
            "Do you have pricing details?",
            "Can I get a quote?",
            "Is pricing customized?",
            "What affects the pricing?",
        ])

    if category == "product":
        if is_recommendation_question(question) and not has_recommendation_requirements(question, conversation_history):
            return _dedupe_suggestions([
                "For customer-facing work",
                "For sales and growth",
                "For internal operations",
                "Not sure yet",
            ])
        if is_list_question(question):
            grounded_items = _extract_grounded_answer_items(answer)
            if grounded_items:
                suggestions = [f"Tell me more about {grounded_items[0]}"]
                if len(grounded_items) > 1:
                    suggestions.append(f"Tell me more about {grounded_items[1]}")
                suggestions.extend([
                    "Which product fits our needs?",
                    "How do these products differ?",
                ])
                return _dedupe_suggestions(suggestions)
            return _dedupe_suggestions([
                "Which product fits our needs?",
                "How do these products differ?",
            ])
        return _dedupe_suggestions([
            "Tell me more about this product",
            "What features does this product include?",
            "How does this product help?",
            "Is this product the right fit for us?",
        ])

    if category == "service":
        if is_recommendation_question(question) and not has_recommendation_requirements(question, conversation_history):
            return _dedupe_suggestions([
                "For setup and onboarding",
                "For integration help",
                "For training and enablement",
                "Not sure yet",
            ])
        if is_list_question(question):
            grounded_items = _extract_grounded_answer_items(answer)
            if grounded_items:
                suggestions = [f"Tell me more about {grounded_items[0]}"]
                if len(grounded_items) > 1:
                    suggestions.append(f"Tell me more about {grounded_items[1]}")
                suggestions.extend([
                    "What is included in each service?",
                    "How do these services differ?",
                ])
                return _dedupe_suggestions(suggestions)
            return _dedupe_suggestions([
                "What is included in each service?",
                "How do these services differ?",
            ])
        return _dedupe_suggestions([
            "Tell me more about this service",
            "What is included in this service?",
            "How does this service work?",
            "Is this service the right fit for us?",
        ])

    if category == "offering":
        if is_recommendation_question(question) and not has_recommendation_requirements(question, conversation_history):
            return _dedupe_suggestions([
                "For customer-facing work",
                "For internal operations",
                "For team enablement",
                "Not sure yet",
            ])
        return _dedupe_suggestions([
            "What are your products?",
            "What are your services?",
            "Which option would you recommend?",
            "How can you help us?",
        ])

    return _dedupe_suggestions([
        "What do you offer?",
        "How can you help?",
        "Tell me more about your company",
        "How do I get started?",
    ])


def _build_suggestions(
    question: str,
    answer: str,
    *,
    agent_doc: dict | None = None,
    intent: str | None = None,
    conversation_history: list[dict] | None = None,
) -> list[str]:
    if intent in {"greeting", "conversation_restart", "identity", "help_request"}:
        return _starter_suggestions(agent_doc)

    category, pricing_subject = _infer_suggestion_topic(
        question,
        answer,
        conversation_history=conversation_history,
    )
    suggestions = _topic_suggestions(
        category,
        question=question,
        answer=answer,
        conversation_history=conversation_history,
        pricing_subject=pricing_subject,
    )

    if is_fallback_response(answer) and not suggestions:
        return _starter_suggestions(agent_doc)

    return _dedupe_suggestions(suggestions or _starter_suggestions(agent_doc))


async def _build_response_suggestions(
    question: str,
    answer: str,
    *,
    agent_doc: dict | None = None,
    intent: str | None = None,
    conversation_history: list[dict] | None = None,
    cs: ChunkStore | None = None,
    agent_id: str = "",
) -> list[str]:
    if is_comparison_question(question):
        comparison_suggestions = await _build_comparison_suggestions_for_agent(
            question,
            agent_id,
            cs,
        )
        if comparison_suggestions:
            return comparison_suggestions

    return _build_suggestions(
        question,
        answer,
        agent_doc=agent_doc,
        intent=intent,
        conversation_history=conversation_history,
    )


def handle_intent(intent: str, message: str = "", agent_doc: dict | None = None) -> str:
    """Enhancement 4: pick a random variant so responses never sound canned."""
    if intent == "identity":
        return _identity_response(agent_doc)

    if intent == "greeting":
        matched = _matching_greeting_response(message)
        if matched:
            return matched

    variants = _INTENT_RESPONSES.get(intent)
    if variants:
        return random.choice(variants)
    return "How can I help you?"


def _strip_greeting_prefix(message: str) -> str:
    stripped = _GREETING_PREFIX_RE.sub("", message.strip(), count=1).strip()
    if stripped != message.strip():
        return stripped

    consume_count = _greeting_prefix_token_count(message)
    if consume_count <= 0:
        return message.strip()

    raw_tokens = message.strip().split()
    if consume_count >= len(raw_tokens):
        return ""
    return " ".join(raw_tokens[consume_count:]).strip(" ,.!?:;-")



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

_LIST_EXCLUDE_LABEL_PATTERNS = re.compile(
    r"\b(about|contact|get in touch|terms|privacy|policy|data protection|"
    r"cookie|careers|blog|news|press|documentation|faq|request a demo|demo|"
    r"downloads?|company history|history|leadership|management|our clients|clients|"
    r"locations?|product finder|service finder)\b",
    re.IGNORECASE,
)
_LIST_EXCLUDE_URL_PATTERNS = re.compile(
    r"/(about|contact|legal|terms|privacy|cookie|cookies|data-protection|"
    r"career|careers|blog|blogs|news|press|documentation|docs|faq|faqs|"
    r"industry|industries|request-demo|demo|download|downloads|company|history|"
    r"management|leadership|team|client|clients|location|locations)/",
    re.IGNORECASE,
)

_LARGE_CATALOG_LIST_THRESHOLD = int(os.getenv("LARGE_CATALOG_LIST_THRESHOLD", "80"))
_AGENTS_DATA_ROOT = Path(os.getenv("AGENTS_DATA_ROOT", "./data/agents"))
_CATALOG_PAGE_HINTS = {
    "product": re.compile(r"(?i)\b(product\s*finder|our products|all products|product type|product category)\b"),
    "service": re.compile(r"(?i)\b(our services|all services|service type|service category)\b"),
}
_CATALOG_FILTER_HEADING_HINTS = {
    "product": re.compile(r"(?i)^\s*(?:product\s*type|product\s*category|categories)\s*$"),
    "service": re.compile(r"(?i)^\s*(?:service\s*type|service\s*category|categories)\s*$"),
}
_CATALOG_FILTER_STOP_PATTERN = re.compile(
    r"(?i)^\s*(?:join our email list|discover|about us|contact us|downloads|news room|careers|reach us|"
    r"follow us|resources|view all resources|location finder|find products|find distributors|"
    r"product information|product characteristics|core products|client stories|partnerships|"
    r"related articles|related products|read more|view product|terms(?:\s*&\s*conditions)?|privacy policy|"
    r"filter by:?|email|follow us)\s*$"
)
_CATALOG_FILTER_OPTION_SKIP = {
    "discover",
    "downloads",
    "contact us",
    "about us",
    "careers",
    "email",
    "resources",
    "location finder",
    "product finder",
    "service finder",
}
_CATEGORY_TOKEN_STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "your",
    "our",
    "from",
    "into",
    "type",
    "types",
    "category",
    "categories",
    "products",
    "product",
    "services",
    "service",
}
_CATALOG_GROUP_GENERIC_SEGMENTS = {
    "product", "products", "service", "services", "solution", "solutions",
    "offering", "offerings", "catalog", "categories", "category", "items",
    "platform", "platforms",
}


def _extract_numbered_items(text: str) -> dict[int, str]:
    """
    Parse a numbered list from a bot reply.
    Handles formats:  1. Item Name   or   1) Item Name
    Returns {1: "Item Name", 2: "Item Name", ...}
    """
    items: dict[int, str] = {}
    for m in re.finditer(r"(\d+)[.)]\s+([^\n0-9][^\n]{2,})", text):
        idx = int(m.group(1))
        label = m.group(2).strip().rstrip(".,;:")
        if label:
            items[idx] = label
    return items


def _extract_summary_label(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    if ":" in first_line:
        prefix, value = first_line.split(":", 1)
        if prefix.strip().lower() in {"product", "service"}:
            return value.strip()
    return ""


def _extract_summary_field(content: str, field_name: str) -> str:
    wanted = field_name.strip().lower()
    for line in content.splitlines():
        if ":" not in line:
            continue
        prefix, value = line.split(":", 1)
        if prefix.strip().lower() == wanted:
            return value.strip()
    return ""


def _compact_summary_for_list(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    if ":" in first_line:
        prefix, value = first_line.split(":", 1)
        prefix_clean = prefix.strip().lower()
        if prefix_clean in {"product", "service"}:
            return f"{prefix_clean.title()}: {value.strip()}"
    return first_line or content.strip()


def _humanize_catalog_segment(segment: str) -> str:
    cleaned = re.sub(r"[-_]+", " ", (segment or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def _derive_catalog_group_label(content: str, source_name: str) -> str | None:
    explicit_category = _extract_summary_field(content, "category")
    if explicit_category:
        return explicit_category

    if source_name.startswith("http"):
        try:
            parsed = urlparse(source_name)
            raw_segments = [
                segment.strip().lower()
                for segment in (parsed.path or "").split("/")
                if segment.strip()
            ]
        except Exception:
            raw_segments = []
        usable_segments = [
            segment
            for segment in raw_segments
            if segment not in _CATALOG_GROUP_GENERIC_SEGMENTS
            and not segment.isdigit()
            and not re.fullmatch(r"\d{4}", segment)
        ]
        if len(usable_segments) >= 2:
            group_label = _humanize_catalog_segment(usable_segments[0])
            item_label = _normalize_summary_label(_extract_summary_label(content))
            if group_label and _normalize_summary_label(group_label) != item_label:
                return group_label

    summary_label = _extract_summary_label(content)
    if not summary_label:
        return None

    separator_match = re.split(r"\s(?:[-|:>])\s", summary_label, maxsplit=1)
    if len(separator_match) == 2:
        candidate = separator_match[0].strip()
        if 1 <= len(candidate.split()) <= 5:
            normalized_candidate = _normalize_summary_label(candidate)
            if normalized_candidate and normalized_candidate != _normalize_summary_label(summary_label):
                return candidate
    return None


def _build_catalog_groups(
    summary_chunks: list[dict],
    *,
    list_category: str,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for chunk in summary_chunks:
        content = str(chunk.get("content") or "").strip()
        source_name = str(chunk.get("source_name") or "unknown")
        if not content or _should_skip_summary_for_list(content, list_category):
            continue
        group_label = _derive_catalog_group_label(content, source_name)
        if not group_label:
            continue
        item_label = _extract_summary_label(content) or _compact_summary_for_list(content)
        grouped.setdefault(group_label, []).append({
            "content": f"{list_category.title()}: {item_label}",
            "source_name": source_name,
            "item_label": item_label,
        })

    groups = [
        {
            "label": label,
            "count": len(items),
            "items": items,
        }
        for label, items in grouped.items()
        if len(items) >= 2
    ]
    groups.sort(key=lambda group: (-group["count"], group["label"].lower()))
    return groups


def _looks_like_catalog_filter_option(line: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (line or "").strip())
    if not cleaned or len(cleaned) > 60:
        return False
    if cleaned.lower() in _CATALOG_FILTER_OPTION_SKIP:
        return False
    if "@" in cleaned or cleaned.startswith("http"):
        return False
    if any(ch.isdigit() for ch in cleaned):
        return False
    if cleaned.endswith(":"):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z&/+.-]*", cleaned)
    return 1 <= len(words) <= 6


def _extract_catalog_filter_options(text: str, *, list_category: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return []

    heading_pattern = _CATALOG_FILTER_HEADING_HINTS.get(list_category)
    start_index: int | None = None
    for idx, line in enumerate(lines[:120]):
        if heading_pattern and heading_pattern.match(line):
            start_index = idx + 1
            break
        if line.lower().startswith("filter by"):
            for look_ahead in range(idx + 1, min(idx + 8, len(lines))):
                if heading_pattern and heading_pattern.match(lines[look_ahead]):
                    start_index = look_ahead + 1
                    break
            if start_index is not None:
                break

    if start_index is None:
        return []

    options: list[str] = []
    seen: set[str] = set()
    started = False
    for line in lines[start_index:]:
        if _CATALOG_FILTER_STOP_PATTERN.match(line):
            if started:
                break
            continue
        if not _looks_like_catalog_filter_option(line):
            if started and len(options) >= 3:
                break
            continue
        started = True
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        options.append(line)
    return options


def _get_agent_website_catalog_categories(agent_id: str, *, list_category: str) -> list[str]:
    website_dir = _AGENTS_DATA_ROOT / agent_id / "websites"
    if not website_dir.exists():
        return []

    categories: list[str] = []
    seen: set[str] = set()
    page_hint = _CATALOG_PAGE_HINTS.get(list_category)
    heading_hint = _CATALOG_FILTER_HEADING_HINTS.get(list_category)

    for path in sorted(website_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for page in payload.get("pages", []) or []:
            url = str(page.get("url") or "")
            title = str(page.get("title") or "")
            text = str(page.get("text") or "")
            text_window = text[:4000]
            is_catalog_landing_page = (
                (list_category == "product" and re.search(r"/products?$", url))
                or (list_category == "service" and re.search(r"/services?$", url))
                or (page_hint is not None and page_hint.search(title))
                or (
                    "filter by" in text_window.lower()
                    and heading_hint is not None
                    and heading_hint.search(text_window)
                )
            )
            if not is_catalog_landing_page:
                continue
            for category in _extract_catalog_filter_options(text, list_category=list_category):
                normalized = category.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                categories.append(category)
    return categories


def _category_tokens(label: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z]+", (label or "").lower()):
        token = raw[:-1] if raw.endswith("s") and len(raw) > 4 else raw
        if len(token) < 3 or token in _CATEGORY_TOKEN_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _filter_summary_chunks_by_category_hint(
    summary_chunks: list[dict],
    *,
    category_label: str,
    list_category: str,
) -> list[dict]:
    tokens = _category_tokens(category_label)
    if not tokens:
        return []

    minimum_hits = 1 if len(tokens) == 1 else 2
    matched: list[dict] = []
    for chunk in summary_chunks:
        content = str(chunk.get("content") or "").strip()
        if not content or _should_skip_summary_for_list(content, list_category):
            continue
        haystack = content.lower()
        hit_count = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", haystack))
        if hit_count >= minimum_hits:
            matched.append({
                "content": f"{list_category.title()}: {_extract_summary_label(content) or _compact_summary_for_list(content)}",
                "source_name": str(chunk.get("source_name") or "unknown"),
                "item_label": _extract_summary_label(content) or _compact_summary_for_list(content),
            })
    return matched


def _build_website_catalog_groups(
    agent_id: str,
    *,
    list_category: str,
    summary_chunks: list[dict],
) -> list[dict]:
    categories = _get_agent_website_catalog_categories(agent_id, list_category=list_category)
    groups: list[dict] = []
    for category in categories:
        items = _filter_summary_chunks_by_category_hint(
            summary_chunks,
            category_label=category,
            list_category=list_category,
        )
        groups.append({
            "label": category,
            "count": len(items),
            "items": items,
        })
    return groups


def _match_catalog_group(question: str, groups: list[dict]) -> dict | None:
    lower = (question or "").lower().strip()
    if not lower:
        return None
    for group in groups:
        label = str(group.get("label") or "").strip()
        if not label:
            continue
        label_lower = label.lower()
        if lower == label_lower or re.search(rf"\b{re.escape(label_lower)}\b", lower):
            return group
    return None


def _extract_summary_url(content: str) -> str:
    for line in content.splitlines():
        if line.lower().startswith("url:"):
            return line.split(":", 1)[1].strip()
    return ""


def _normalize_summary_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _normalize_compare_text(value: str) -> str:
    lowered = (value or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _compact_compare_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _extract_comparison_targets(question: str) -> list[str]:
    normalized = " ".join((question or "").strip().split())
    if not normalized:
        return []

    patterns = (
        r"(?i)\bcompare\s+(.+?)\s+(?:vs|versus)\s+(.+)$",
        r"(?i)\bcompare\s+(.+?)\s+and\s+(.+)$",
        r"(?i)\b(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+)$",
        r"(?i)\b(.+?)\s+(?:vs|versus)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        targets = []
        for group in match.groups():
            candidate = str(group or "").strip(" .,:;!?-")
            candidate = re.sub(r"(?i)\bproducts?\b$", "", candidate).strip(" .,:;!?-")
            if candidate:
                targets.append(candidate)
        if len(targets) >= 2:
            return targets[:2]
    return []


def _load_agent_website_pages(agent_id: str) -> list[tuple[str, str, str]]:
    website_dir = _AGENTS_DATA_ROOT / agent_id / "websites"
    if not website_dir.exists():
        return []

    pages: list[tuple[str, str, str]] = []
    for path in sorted(website_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for page in payload.get("pages", []) or []:
            url = str(page.get("url") or "").strip()
            title = str(page.get("title") or "")
            text = str(page.get("text") or "")
            if not url or not text:
                continue
            pages.append((url, title, text))
    return pages


def _build_agent_website_category_map(agent_id: str, *, list_category: str) -> dict[str, str]:
    if list_category not in {"product", "service"}:
        return {}

    pages = _load_agent_website_pages(agent_id)
    if not pages:
        return {}

    try:
        from app.chunking import extract_site_catalog_categories, _match_catalog_category_hint
    except Exception:
        return {}

    catalog_categories = extract_site_catalog_categories(pages, category=list_category)
    if not catalog_categories:
        return {}

    mapping: dict[str, str] = {}
    for url, title, text in pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        hint = _match_catalog_category_hint(
            title=title,
            url=url,
            lines=lines,
            category=list_category,
            catalog_categories=catalog_categories,
        )
        if hint:
            mapping[_normalize_website_url_for_match(url)] = hint
    return mapping


def _extract_page_compare_labels(url: str, title: str, text: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add(label: str | None) -> None:
        value = " ".join(str(label or "").split()).strip(" -|.,;:")
        if not value:
            return
        normalized = _normalize_compare_text(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        labels.append(value)

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    title_clean = str(title or "").strip()
    if title_clean:
        add(title_clean)
    for line in lines[:8]:
        add(line)

    try:
        parsed = urlparse(url or "")
        slug = parsed.path.rstrip("/").split("/")[-1]
        if slug:
            add(re.sub(r"[-_]+", " ", slug))
    except Exception:
        pass

    return labels


def _resolve_comparison_targets_from_website(
    question: str,
    agent_id: str,
) -> list[dict[str, str]]:
    targets = _extract_comparison_targets(question)
    if len(targets) < 2:
        return []

    pages = _load_agent_website_pages(agent_id)
    if not pages:
        return []

    category_maps = {
        "product": _build_agent_website_category_map(agent_id, list_category="product"),
        "service": _build_agent_website_category_map(agent_id, list_category="service"),
    }

    resolved: list[dict[str, str]] = []
    used_sources: set[str] = set()
    for target in targets:
        normalized_target = _normalize_compare_text(target)
        compact_target = _compact_compare_text(target)
        if not normalized_target:
            return []

        best_match: dict[str, str] | None = None
        best_score = -1
        for url, title, text in pages:
            normalized_url = _normalize_website_url_for_match(url)
            labels = _extract_page_compare_labels(url, title, text)
            if not labels:
                continue
            for label in labels:
                normalized_label = _normalize_compare_text(label)
                compact_label = _compact_compare_text(label)
                score = -1
                if compact_target and compact_target == compact_label:
                    score = 300 + len(compact_label)
                elif normalized_target and normalized_target == normalized_label:
                    score = 260 + len(normalized_label)
                elif compact_target and compact_target in compact_label:
                    score = 220 + len(compact_target)
                else:
                    target_tokens = [tok for tok in normalized_target.split() if tok]
                    if target_tokens and all(re.search(rf"\b{re.escape(tok)}\b", normalized_label) for tok in target_tokens):
                        score = 160 + len(target_tokens)
                if score <= best_score:
                    continue

                chunk_category = ""
                main_category = ""
                for candidate_category in ("product", "service"):
                    mapped = category_maps[candidate_category].get(normalized_url, "")
                    if mapped:
                        chunk_category = candidate_category
                        main_category = mapped
                        break
                if not chunk_category:
                    continue

                best_score = score
                best_match = {
                    "label": label,
                    "main_category": main_category,
                    "source_name": url,
                    "chunk_category": chunk_category,
                    "question_position": str(_question_label_position(question, target)),
                }

        if not best_match:
            return []
        source_key = _normalize_website_url_for_match(best_match["source_name"])
        if source_key in used_sources:
            continue
        used_sources.add(source_key)
        resolved.append(best_match)

    return resolved if len(resolved) >= 2 else []


def _label_match_tokens(label: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", (label or "").lower()):
        if len(token) <= 1:
            continue
        if token in {"product", "service", "solution", "solutions"}:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _question_mentions_summary_label(question: str, label: str) -> bool:
    normalized_question = _normalize_summary_label(question)
    normalized_label = _normalize_summary_label(label)
    if not normalized_question or not normalized_label:
        return False
    if normalized_label in normalized_question:
        return True

    tokens = _label_match_tokens(label)
    if not tokens:
        return False
    return all(re.search(rf"\b{re.escape(token)}\b", normalized_question) for token in tokens)


def _question_label_position(question: str, label: str) -> int:
    normalized_question = _normalize_summary_label(question)
    normalized_label = _normalize_summary_label(label)
    if not normalized_question or not normalized_label:
        return 10**9

    direct = normalized_question.find(normalized_label)
    if direct >= 0:
        return direct

    positions: list[int] = []
    for token in _label_match_tokens(label):
        match = re.search(rf"\b{re.escape(token)}\b", normalized_question)
        if match:
            positions.append(match.start())
    return min(positions) if positions else 10**9


def _extract_same_category_comparison_scope(
    question: str,
    summary_chunks: list[dict],
    *,
    agent_id: str = "",
) -> dict[str, object]:
    website_category_maps: dict[str, dict[str, str]] = {}

    matched_entries: list[dict[str, str]] = []
    for chunk in summary_chunks:
        content = str(chunk.get("content") or "").strip()
        label = _extract_summary_label(content)
        if not label or not _question_mentions_summary_label(question, label):
            continue
        chunk_category = str(chunk.get("category") or "")
        source_name = str(chunk.get("source_name") or "unknown")
        main_category = _extract_summary_field(content, "category")
        if not main_category and agent_id and chunk_category in {"product", "service"}:
            category_map = website_category_maps.get(chunk_category)
            if category_map is None:
                category_map = _build_agent_website_category_map(agent_id, list_category=chunk_category)
                website_category_maps[chunk_category] = category_map
            main_category = category_map.get(_normalize_website_url_for_match(source_name), "")
        if not main_category:
            continue
        matched_entries.append({
            "label": label,
            "main_category": main_category,
            "source_name": source_name,
            "chunk_category": chunk_category,
            "question_position": str(_question_label_position(question, label)),
        })

    matched_entries.sort(
        key=lambda entry: (
            int(entry.get("question_position") or 10**9),
            _normalize_summary_label(entry.get("label", "")),
        )
    )

    if len(matched_entries) < 2 and agent_id:
        website_matches = _resolve_comparison_targets_from_website(question, agent_id)
        if len(website_matches) >= len(matched_entries):
            matched_entries = website_matches

    distinct_categories = {
        _normalize_summary_label(entry["main_category"])
        for entry in matched_entries
        if entry.get("main_category")
    }
    distinct_chunk_categories = {
        str(entry.get("chunk_category") or "")
        for entry in matched_entries
        if entry.get("chunk_category")
    }

    if not matched_entries or len(distinct_categories) != 1 or len(distinct_chunk_categories) != 1:
        return {
            "main_category": None,
            "chunk_category": None,
            "source_names": set(),
            "cross_category": len(distinct_categories) > 1,
            "matched_entries": matched_entries,
            "focus_entry": matched_entries[0] if matched_entries else None,
        }

    main_category = matched_entries[0]["main_category"]
    chunk_category = matched_entries[0]["chunk_category"]
    source_names: set[str] = set()
    category_map = website_category_maps.get(chunk_category) or {}
    if not category_map and agent_id and chunk_category in {"product", "service"}:
        category_map = _build_agent_website_category_map(agent_id, list_category=chunk_category)
        website_category_maps[chunk_category] = category_map
    for chunk in summary_chunks:
        current_chunk_category = str(chunk.get("category") or "")
        if current_chunk_category != chunk_category:
            continue
        source_name = str(chunk.get("source_name") or "unknown")
        chunk_main_category = _extract_summary_field(str(chunk.get("content") or ""), "category")
        if not chunk_main_category and agent_id and category_map:
            chunk_main_category = category_map.get(_normalize_website_url_for_match(source_name), "")
        if _normalize_summary_label(chunk_main_category) == _normalize_summary_label(main_category):
            source_names.add(source_name)
    return {
        "main_category": main_category,
        "chunk_category": chunk_category,
        "source_names": source_names,
        "cross_category": False,
        "matched_entries": matched_entries,
        "focus_entry": matched_entries[0] if matched_entries else None,
    }


def _is_root_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return (parsed.path or "/") == "/"


def _should_skip_summary_for_list(content: str, list_category: str | None) -> bool:
    if not list_category:
        return False

    label = _extract_summary_label(content)
    summary_url = _extract_summary_url(content)

    if label and _LIST_EXCLUDE_LABEL_PATTERNS.search(label):
        return True
    if summary_url and _LIST_EXCLUDE_URL_PATTERNS.search(summary_url):
        return True

    # Root/home pages are often catalog or company-overview pages rather than a
    # single product/service item. Let detail retrieval use them, but don't turn
    # the homepage title itself into a fake product entry in list answers.
    if summary_url and _is_root_url(summary_url):
        return True

    return False


def _should_skip_detail_chunk_for_list(source_name: str, list_category: str | None) -> bool:
    if not list_category:
        return False
    if source_name and _is_root_url(source_name):
        return True
    return bool(source_name and _LIST_EXCLUDE_URL_PATTERNS.search(source_name))


def _list_scope_categories(list_category: str | None) -> list[str]:
    if list_category == "offering":
        return ["product", "service"]
    if list_category in {"product", "service"}:
        return [list_category]
    return []


def _matches_list_scope(chunk_category: str | None, list_category: str | None) -> bool:
    if not list_category:
        return True
    return (chunk_category or "") in _list_scope_categories(list_category)


def _extract_last_category_reference(last_bot_message: str) -> tuple[str | None, str | None]:
    text = str(last_bot_message or "").strip()
    if not text:
        return None, None

    category_patterns = (
        r"(?im)^\s*Product Category\s*:\s*(.+?)\s*$",
        r"(?im)^\s*Service Category\s*:\s*(.+?)\s*$",
        r"(?i)\b([A-Za-z][A-Za-z0-9 &+/\-]{2,}?)\s+category\s+contains\s*\d+\s+(product|products|service|services)\b",
    )
    for pattern in category_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if len(match.groups()) >= 2 and match.group(2):
            raw_label = match.group(1)
            item_type = match.group(2).lower()
        else:
            raw_label = match.group(1)
            item_type = "products" if "product category" in match.group(0).lower() else "services"

        label = " ".join(str(raw_label or "").split()).strip(" -|.,;:")
        if not label:
            continue
        if label.lower().startswith("the "):
            label = label[4:].strip()
        return label, item_type
    return None, None


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
        category_label, category_item_type = _extract_last_category_reference(last_bot_message)
        import logging; logging.getLogger("chatbot").info(f"[RESOLVE] q={repr(question)} lbm_len={len(last_bot_message)} items={list(numbered.keys())}")

        category_followup_patterns = (
            r"\bthat product\b",
            r"\bthat service\b",
            r"\bthat category\b",
            r"\bthis product\b",
            r"\bthis service\b",
            r"\bthis category\b",
            r"\bthose products\b",
            r"\bthose services\b",
            r"\bwhat is that product\b",
            r"\bwhat are those products\b",
            r"\bshow (?:me )?(?:the )?(?:products|services)\b",
            r"\blist (?:the )?(?:products|services)\b",
        )
        if category_label and category_item_type and any(re.search(pattern, q_lower) for pattern in category_followup_patterns):
            noun = "services" if "service" in category_item_type else "products"
            return f"What {noun} are in {category_label}?"

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


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?:(?:\+?\d[\d\s().-]{7,}\d))")
_LEAD_INTENT_RE = re.compile(
    r"\b("
    r"demo|quote|quotation|pricing|price|cost|proposal|rfq|"
    r"contact me|call me|reach me|reach out|follow up|"
    r"callback|consultation|book|schedule|meeting|sales|purchase|buy|"
    r"interested|interested in|showing interest|"
    r"customization|customisation|customize|customise|"
    r"integration|integrations|integrate|"
    r"let'?s talk|talk to someone|get in touch"
    r")\b",
    re.I,
)
_QUALIFIED_INQUIRY_RE = re.compile(
    r"\b("
    r"demo|book a demo|schedule a demo|request a demo|"
    r"quote|quotation|pricing|price|cost|proposal|rfq|"
    r"service request|support request|implementation request|"
    r"customization|customisation|customize|customise|"
    r"integration|integrations|integrate|"
    r"inspection|assessment|audit|site visit|"
    r"erp customization|erp customisation|erp inspection|erp assessment|erp audit|"
    r"interested|interested in|showing interest|buy|purchase|get in touch|contact me|reach out"
    r")\b",
    re.I,
)
_CONTACT_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"[\d\s().+-]{7,}|"
    r"my email is .*|my phone is .*|you can reach me .*"
    r")\s*$",
    re.I,
)
_BUSINESS_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_BUSINESS_PHONE_RE = re.compile(r"(?:\+\d[\d\s().-]{7,}\d|\b\d{7,15}\b)")


def _clean_lead_value(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned.strip(" ,.;:-") or None


def _compact_contact_chunk(content: str) -> str | None:
    text = str(content or "").strip()
    if not text:
        return None
    lower = text.lower()
    has_email_cue = any(token in lower for token in ("email", "e:", "e-mail", "mail"))
    has_phone_cue = any(token in lower for token in ("phone", "mobile", "tel", "t:", "call"))

    emails: list[str] = []
    seen_emails: set[str] = set()
    for match in _BUSINESS_EMAIL_RE.finditer(text):
        email = match.group(0).strip()
        key = email.lower()
        if key in seen_emails:
            continue
        seen_emails.add(key)
        emails.append(email)

    phones: list[str] = []
    seen_phones: set[str] = set()
    for match in _BUSINESS_PHONE_RE.finditer(text):
        phone = match.group(0).strip()
        digits = re.sub(r"\D", "", phone)
        if len(digits) < 7:
            continue
        if digits in seen_phones:
            continue
        seen_phones.add(digits)
        phones.append(phone)

    if not emails and not phones:
        return None
    if emails and not phones and not has_email_cue:
        return None

    lines = ["Sales Contact:"]
    lines.extend(f"Email: {email}" for email in emails[:2])
    lines.extend(f"Phone: {phone}" for phone in phones[:2])
    return "\n".join(lines)


def _is_compact_contact_submission(text: str) -> bool:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return False

    email_present = bool(_EMAIL_RE.search(normalized))
    digit_groups = re.findall(r"\b\d[\d\s().+-]{4,}\d\b|\b\d{6,15}\b", normalized)
    word_count = len(re.findall(r"[A-Za-z0-9@._%+-]+", normalized))

    return word_count <= 8 and (email_present or bool(digit_groups))


def _extract_email(text: str) -> str | None:
    match = _EMAIL_RE.search(text or "")
    return match.group(0).lower() if match else None


def _extract_phone(text: str) -> str | None:
    for match in _PHONE_RE.finditer(text or ""):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if len(digits) >= 7:
            return candidate

    if _is_compact_contact_submission(text):
        for match in re.finditer(r"\b\d{6,15}\b", text or ""):
            candidate = match.group(0).strip()
            if len(candidate) >= 6:
                return candidate
    return None


def _looks_like_name(value: str) -> bool:
    lower = value.lower().strip()
    if not lower:
        return False
    blocked_prefixes = (
        "interested", "looking", "here", "ready", "need", "want", "from",
        "available", "contact", "email", "phone", "number", "quote", "demo",
    )
    if any(lower.startswith(prefix) for prefix in blocked_prefixes):
        return False
    words = [word for word in re.split(r"\s+", value.strip()) if word]
    if len(words) > 4:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", value.strip()))


def _sanitize_name_candidate(value: str | None) -> str | None:
    candidate = _clean_lead_value(value)
    if not candidate:
        return None

    candidate = re.split(
        r"(?i)\b(?:my email is|email is|you can reach me|reach me at|phone is|mobile is|contact me at|from)\b",
        candidate,
        maxsplit=1,
    )[0]
    candidate = re.split(r"[,.!;:\n]", candidate, maxsplit=1)[0]
    candidate = _clean_lead_value(candidate)

    if candidate and _looks_like_name(candidate):
        return candidate
    return None


def _extract_name(text: str) -> str | None:
    patterns = (
        r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{1,80})",
        r"\bi am\s+([A-Za-z][A-Za-z .'-]{1,80})",
        r"\bi'm\s+([A-Za-z][A-Za-z .'-]{1,80})",
        r"\bthis is\s+([A-Za-z][A-Za-z .'-]{1,80})",
        r"\bname\s*:\s*([A-Za-z][A-Za-z .'-]{1,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if not match:
            continue
        candidate = _sanitize_name_candidate(match.group(1))
        if candidate:
            return candidate

    if _is_compact_contact_submission(text):
        stripped = _EMAIL_RE.sub(" ", text or "")
        stripped = _PHONE_RE.sub(" ", stripped)
        stripped = re.sub(r"\b\d{6,15}\b", " ", stripped)
        stripped = re.sub(r"[^A-Za-z .'-]", " ", stripped)
        stripped = " ".join(stripped.split())
        candidate = _sanitize_name_candidate(stripped)
        if candidate:
            return candidate
    return None


def _extract_company(text: str) -> str | None:
    patterns = (
        r"\b(?:company|organization|organisation|business)\s*(?:name)?\s*(?:is|:)\s*([A-Za-z0-9][A-Za-z0-9 &.,'-]{1,80})",
        r"\b(?:i am|i'm)\s+from\s+([A-Za-z0-9][A-Za-z0-9 &.,'-]{1,80})",
        r"\b(?:we are|we're)\s+from\s+([A-Za-z0-9][A-Za-z0-9 &.,'-]{1,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if not match:
            continue
        candidate = _clean_lead_value(match.group(1))
        if candidate and len(candidate) >= 2:
            return candidate
    return None


def _extract_interest(user_messages: list[str]) -> str | None:
    if not user_messages:
        return None
    for text in reversed(user_messages):
        cleaned = _clean_lead_value(_EMAIL_RE.sub("", _PHONE_RE.sub("", text)))
        if not cleaned or _CONTACT_ONLY_RE.match(cleaned):
            continue
        if _LEAD_INTENT_RE.search(cleaned):
            return cleaned[:220]
    for text in reversed(user_messages):
        cleaned = _clean_lead_value(_EMAIL_RE.sub("", _PHONE_RE.sub("", text)))
        if cleaned and len(cleaned.split()) >= 4 and not _CONTACT_ONLY_RE.match(cleaned):
            return cleaned[:220]
    return None


def _has_qualified_inquiry(user_messages: list[str]) -> bool:
    for text in user_messages:
        cleaned = _clean_lead_value(_EMAIL_RE.sub("", _PHONE_RE.sub("", text)))
        if cleaned and _QUALIFIED_INQUIRY_RE.search(cleaned):
            return True
    return False


def _conversation_has_qualified_inquiry(
    question: str,
    conversation_history: list[dict] | None = None,
) -> bool:
    if _QUALIFIED_INQUIRY_RE.search(question or ""):
        return True
    if not conversation_history:
        return False
    user_messages = [
        str(msg.get("content") or "")
        for msg in conversation_history
        if msg.get("role") == "user"
    ]
    return _has_qualified_inquiry(user_messages)


async def _maybe_capture_lead(
    *,
    store: AdminStoreMongo,
    settings: dict,
    tenant_id: str,
    agent_id: str,
    conversation: dict,
    source: str,
) -> None:
    if not settings.get("lead_capture_enabled"):
        return

    messages = conversation.get("messages", [])
    user_messages = [
        str(msg.get("content") or "").strip()
        for msg in messages
        if msg.get("role") == "user" and str(msg.get("content") or "").strip()
    ]
    if not user_messages:
        return

    email = None
    phone = None
    name = None
    company = None
    for text in user_messages:
        email = email or _extract_email(text)
        phone = phone or _extract_phone(text)
        name = name or _extract_name(text)
        company = company or _extract_company(text)

    qualified_inquiry = _has_qualified_inquiry(user_messages)
    if not email and not phone and not qualified_inquiry:
        return

    interest = _extract_interest(user_messages)
    await store.upsert_lead(
        tenant_id=tenant_id,
        agent_id=agent_id,
        conversation_id=conversation["id"],
        source=source,
        name=name,
        email=email,
        phone=phone,
        company=company,
        interest=interest,
        notes=interest,
    )


async def _build_context(
    pipeline: RAGPipeline,
    question: str,
    cs: ChunkStore,
    loop: asyncio.AbstractEventLoop,
    agent_id: str = "",
    conversation_history: list[dict] | None = None,
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
    # ── Step 1a: fetch summary chunks from MongoDB for list questions ─────────
    is_list_query = is_list_question(question)
    is_comparison_query = is_comparison_question(question)
    list_category = detect_list_category(question) if is_list_query else None

    scoped_categories = _list_scope_categories(list_category)
    summary_context: list[dict] = []
    comparison_detail_context: list[dict] = []
    raw_summary_context: list[dict] = []
    comparison_scope: dict[str, object] = {
        "main_category": None,
        "chunk_category": None,
        "source_names": set(),
        "cross_category": False,
    }
    seen_summary_labels: set[str] = set()
    if agent_id and (is_list_query or is_comparison_query):
        summary_ids = await cs.get_summary_chunk_ids_by_agent(
            agent_id,
            category=list_category if list_category in {"product", "service"} else None,
        )
        if summary_ids:
            summary_chunks = await cs.get_chunks_by_ids(summary_ids)
            for c in summary_chunks:
                if is_list_query and not _matches_list_scope(c.get("category"), list_category):
                    continue
                content = c.get("content", "").strip()
                if is_list_query and _should_skip_summary_for_list(content, list_category):
                    continue
                summary_label = _normalize_summary_label(_extract_summary_label(content))
                if summary_label:
                    if summary_label in seen_summary_labels:
                        continue
                    seen_summary_labels.add(summary_label)
                if content:
                    raw_summary_context.append({
                        "content": content,
                        "source_name": c.get("source_name", "unknown"),
                        "category": c.get("category"),
                    })

            if is_comparison_query and raw_summary_context:
                comparison_scope = _extract_same_category_comparison_scope(
                    question,
                    raw_summary_context,
                    agent_id=agent_id,
                )
                comparison_source_names = comparison_scope.get("source_names") or set()
                comparison_category = str(comparison_scope.get("main_category") or "").strip()
                if comparison_scope.get("cross_category"):
                    summary_context = [{
                        "content": "Comparison Scope: The explicitly referenced items belong to different categories. Ask the user which category they would like to compare within, and do not compare across categories.",
                        "source_name": "comparison",
                    }]
                elif comparison_source_names:
                    summary_context = (
                        ([{
                            "content": f"Comparison Category: {comparison_category}",
                            "source_name": "comparison",
                        }] if comparison_category else [])
                        + [{
                            "content": chunk["content"],
                            "source_name": chunk["source_name"],
                        }
                        for chunk in raw_summary_context
                        if chunk["source_name"] in comparison_source_names]
                    )[:8]
                    comparison_detail_chunks = await cs.get_detail_chunks_by_agent_and_sources(
                        agent_id,
                        source_names=sorted(comparison_source_names),
                        limit=16,
                    )
                    seen_comparison_detail: set[str] = set()
                    for chunk in comparison_detail_chunks:
                        content = str(chunk.get("content") or "").strip()
                        if not content or content in seen_comparison_detail:
                            continue
                        seen_comparison_detail.add(content)
                        comparison_detail_context.append({
                            "content": content,
                            "source_name": str(chunk.get("source_name") or "unknown"),
                        })

    should_group_large_catalog = (
        is_list_query
        and list_category in {"product", "service"}
        and len(raw_summary_context) >= _LARGE_CATALOG_LIST_THRESHOLD
    )
    if should_group_large_catalog:
        catalog_groups = _build_catalog_groups(raw_summary_context, list_category=list_category or "product")
        grouped_item_count = sum(group["count"] for group in catalog_groups)
        use_grouping = (
            len(catalog_groups) >= 3
            and grouped_item_count >= max(12, int(len(raw_summary_context) * 0.6))
        )
        if use_grouping:
            matched_group = _match_catalog_group(question, catalog_groups)
            if matched_group:
                summary_context = [
                    {
                        "content": item["content"],
                        "source_name": item["source_name"],
                    }
                    for item in matched_group["items"]
                ]
            else:
                category_label = "Product Category" if list_category == "product" else "Service Category"
                summary_context = [
                    {
                        "content": f"{category_label}: {group['label']}\nCount: {group['count']}",
                        "source_name": "catalog",
                    }
                    for group in catalog_groups
                ]
        else:
            website_catalog_groups = (
                _build_website_catalog_groups(
                    agent_id,
                    list_category=list_category or "product",
                    summary_chunks=raw_summary_context,
                )
                if agent_id
                else []
            )
            if len(website_catalog_groups) >= 3:
                matched_group = _match_catalog_group(question, website_catalog_groups)
                if matched_group and len(matched_group.get("items") or []) >= 2:
                    summary_context = [
                        {
                            "content": item["content"],
                            "source_name": item["source_name"],
                        }
                        for item in matched_group["items"]
                    ]
                else:
                    category_label = "Product Category" if list_category == "product" else "Service Category"
                    summary_context = [
                        {
                            "content": (
                                f"{category_label}: {group['label']}\nCount: {group['count']}"
                                if group["count"] > 0
                                else f"{category_label}: {group['label']}"
                            ),
                            "source_name": "catalog",
                        }
                        for group in website_catalog_groups
                    ]
            else:
                summary_context = [
                    {
                        "content": _compact_summary_for_list(chunk["content"]),
                        "source_name": chunk["source_name"],
                    }
                    for chunk in raw_summary_context
                ]
    else:
        summary_context = [
            {
                "content": _compact_summary_for_list(chunk["content"]) if is_list_query else chunk["content"],
                "source_name": chunk["source_name"],
            }
            for chunk in raw_summary_context
        ]

    # ── Step 1b: pin text_snippet and QA chunks for list questions ───────────
    # Manually added text snippets and Q&A entries have no summary chunk and
    # their source_name is a plain title — not a URL — so they never appear in
    # step 1a even with the URL filter removed.  We always pin them here so
    # user-curated knowledge (e.g. "Asset Management System") is NEVER silently
    # dropped from listing answers regardless of vector scores.
    # NOTE: Files (PDFs, DOCX, etc.) and general website pages are covered by
    # the vector search in step 2 which uses a 3× retrieval limit + lower
    # threshold for list questions, ensuring broad knowledge base coverage.
    if agent_id and is_list_query and list_category is None:
        manual_chunks = await cs.get_text_snippet_and_qa_chunks_by_agent(agent_id)
        seen_manual: set[str] = {c["content"] for c in summary_context}
        for c in manual_chunks:
            chunk_content = c.get("content", "").strip()
            if chunk_content and chunk_content not in seen_manual:
                seen_manual.add(chunk_content)
                summary_context.append({
                    "content": chunk_content,
                    "source_name": c.get("source_name", "unknown"),
                })

    # Step 1c: pin category-matching non-website detail chunks only when the
    # summary catalog is sparse. If we already have the list of item labels,
    # extra detail paragraphs mostly just bloat the prompt and slow list answers.
    pinned_category_details: list[dict] = []
    if agent_id and is_list_query and scoped_categories and len(summary_context) < 2:
        seen_pinned: set[str] = {c["content"] for c in summary_context}
        for category_name in scoped_categories:
            curated_category_details = await cs.get_detail_chunks_by_agent_and_category(
                agent_id,
                category=category_name,
                limit=8,
                exclude_source_types=["website"],
            )
            for c in curated_category_details:
                content = str(c.get("content") or "").strip()
                source_name = str(c.get("source_name") or "unknown")
                if not content or content in seen_pinned:
                    continue
                if _should_skip_detail_chunk_for_list(source_name, list_category):
                    continue
                seen_pinned.add(content)
                pinned_category_details.append({
                    "content": content,
                    "source_name": source_name,
                })

    # Step 1d: stable website fallback for product/service lists.
    # If summary chunks are missing or too sparse, pin website detail chunks
    # from Mongo so list answers do not depend on a single vector hit.
    if agent_id and is_list_query and scoped_categories and len(summary_context) < 2:
        seen_pinned: set[str] = {c["content"] for c in summary_context}
        seen_pinned.update(c["content"] for c in pinned_category_details)
        per_category_limit = 10 if len(scoped_categories) > 1 else 20
        for category_name in scoped_categories:
            website_category_details = await cs.get_detail_chunks_by_agent_and_category(
                agent_id,
                category=category_name,
                limit=per_category_limit,
                source_types=["website"],
            )
            for c in website_category_details:
                content = str(c.get("content") or "").strip()
                source_name = str(c.get("source_name") or "unknown")
                if not content or content in seen_pinned:
                    continue
                if _should_skip_detail_chunk_for_list(source_name, list_category):
                    continue
                seen_pinned.add(content)
                pinned_category_details.append({
                    "content": content,
                    "source_name": source_name,
                })

    # ── Step 2: normal vector search for detail chunks ────────────────────────
    # For explicit product/service listing questions, summary chunks are the
    # canonical list representation. Once we already have summary-based list
    # context (or pinned category detail fallback), extra vector detail chunks
    # tend to introduce noisy non-item pages like homepages into the list.
    should_include_vector_details = (
        list_category is None
        or (not summary_context and not pinned_category_details)
    )

    detail_context: list[dict] = []
    if should_include_vector_details:
        new_style_ids, legacy_chunks = await loop.run_in_executor(
            None, lambda: pipeline.search_chunks(question)
        )
        comparison_source_names = comparison_scope.get("source_names") or set()
        detail_context = [] if list_category else [
            chunk
            for chunk in legacy_chunks
            if not comparison_source_names or chunk.get("source_name") in comparison_source_names
        ]
        seen_content: set[str] = {c["content"] for c in summary_context}
        seen_content.update(c["content"] for c in comparison_detail_context)
        seen_content.update(c["content"] for c in pinned_category_details)
        if new_style_ids:
            mongo_chunks = await cs.get_chunks_by_ids(new_style_ids)
            for c in mongo_chunks:
                if not _matches_list_scope(c.get("category"), list_category):
                    continue
                source_name = c.get("source_name", "unknown")
                if comparison_source_names and source_name not in comparison_source_names:
                    continue
                if _should_skip_detail_chunk_for_list(str(source_name), list_category):
                    continue
                content = c.get("content", "").strip()
                # Skip detail chunks that duplicate a summary chunk already added
                if content and content not in seen_content:
                    detail_context.append({
                        "content": content,
                        "source_name": source_name,
                    })

    # Summary chunks first, then pinned category details, then vector details.
    contact_context: list[dict] = []
    should_pin_contact_details = (
        bool(agent_id)
        and (
            is_pricing_question(question)
            or (
                is_recommendation_question(question)
                and has_recommendation_requirements(question, conversation_history)
            )
            or _conversation_has_qualified_inquiry(question, conversation_history)
        )
    )
    if should_pin_contact_details:
        seen_contact_content: set[str] = {c["content"] for c in summary_context}
        seen_contact_content.update(c["content"] for c in comparison_detail_context)
        seen_contact_content.update(c["content"] for c in pinned_category_details)
        seen_contact_content.update(c["content"] for c in detail_context)
        contact_chunks = await cs.get_contact_chunks_by_agent(agent_id, limit=4)
        for c in contact_chunks:
            content = _compact_contact_chunk(str(c.get("content") or "").strip())
            if not content or content in seen_contact_content:
                continue
            seen_contact_content.add(content)
            contact_context.append({
                "content": content,
                "source_name": c.get("source_name", "unknown"),
            })

    # Summary chunks first, then comparison-specific detail, then pinned category details,
    # then compact contact chunks for grounded handoff details, and finally vector details.
    return summary_context + comparison_detail_context + pinned_category_details + contact_context + detail_context


async def route_to_llm(
    question: str,
    pipeline: RAGPipeline,
    settings: dict,
    loop: asyncio.AbstractEventLoop,
    conversation_history: list[dict] | None = None,
    chunk_store: ChunkStore | None = None,
    agent_id: str = "",
) -> str:
    offering_scope = None
    if chunk_store is not None:
        context = await _build_context(
            pipeline,
            question,
            chunk_store,
            loop,
            agent_id=agent_id,
            conversation_history=conversation_history,
        )
        if agent_id:
            offering_scope = await chunk_store.get_offering_scope_by_agent(agent_id)
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
            lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
            offering_scope=offering_scope,
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
    suggestions: list[str] = Field(default_factory=list)


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

    raw_question = request.question.strip()
    question = _strip_greeting_prefix(raw_question) or raw_question

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
        answer = handle_intent(intent, question, agent_doc=agent_doc)
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

    answer = _apply_required_lead_followup(
        answer,
        question=question,
        conversation_history=conversation_history,
        lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
    )

    # Enhancement 7: accurate token counting via tiktoken
    input_tokens   = _count_tokens(raw_question)
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
                question=raw_question,
                conversation_id=request.conversation_id,
            )
        except Exception:
            pass

    conversation = await store.append_conversation_messages(
        agent_id=request.agent_id,
        tenant_id=tenant_id,
        user_id=user.id,
        user_message=raw_question,
        assistant_message=answer,
        conversation_id=request.conversation_id,
    )
    try:
        await _maybe_capture_lead(
            store=store,
            settings=settings,
            tenant_id=tenant_id,
            agent_id=request.agent_id,
            conversation=conversation,
            source="chat",
        )
    except Exception:
        pass
    suggestions = await _build_response_suggestions(
        question,
        answer,
        agent_doc=agent_doc,
        intent=intent,
        conversation_history=conversation_history,
        cs=cs,
        agent_id=request.agent_id,
    )
    return ChatResponse(answer=answer, conversation_id=conversation["id"], suggestions=suggestions)


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

    raw_question = request.question.strip()
    question = _strip_greeting_prefix(raw_question) or raw_question

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

    if not last_bot_message and request.last_bot_message and request.last_bot_message.strip():
        last_bot_message = request.last_bot_message.strip()

    if not last_bot_message:
        try:
            recent = await store.db.conversations.find_one(
                {"agent_id": request.agent_id, "tenant_id": tenant_id},
                sort=[("updated_at", -1)],
            )
            if recent:
                recent_msgs = recent.get("messages", [])
                last_bot_message = _get_last_bot_message(recent_msgs)
                if not conversation_history:
                    conversation_history = _get_conversation_history(recent_msgs, max_turns=3)
        except Exception:
            pass

    question = resolve_reference_query(question, last_bot_message)

    intent = detect_intent(question, last_bot_message)
    if intent == "followup_affirmation" and last_bot_message:
        intent = None

    loop = asyncio.get_event_loop()
    pipeline = None if intent else _build_pipeline(store, request.agent_id)
    context = []
    offering_scope = await cs.get_offering_scope_by_agent(request.agent_id)
    if pipeline is not None:
        context = await _build_context(
            pipeline,
            question,
            cs,
            loop,
            agent_id=request.agent_id,
            conversation_history=conversation_history,
        )
    full_answer_parts: list[str] = []

    async def token_stream():
        try:
            if intent:
                answer_text = handle_intent(intent, question, agent_doc=agent_doc)
                full_answer_parts.append(answer_text)
                yield _sse_data(answer_text.replace("\n", "\\n"))
            else:
                gen = await loop.run_in_executor(
                    None,
                    lambda: pipeline.stream_answer_question(
                        question,
                        system_prompt=settings.get("system_prompt"),
                        temperature=settings.get("temperature", 0.2),
                        conversation_history=conversation_history,
                        prefetched_context=context,
                        lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
                        offering_scope=offering_scope,
                    ),
                )
                for token in gen:
                    full_answer_parts.append(token)
                    yield _sse_data(token.replace("\n", "\\n"))
        except Exception as exc:
            yield _sse_event("error", str(exc))
        finally:
            full_answer = "".join(full_answer_parts)
            if full_answer:
                adjusted_answer = _apply_required_lead_followup(
                    full_answer,
                    question=question,
                    conversation_history=conversation_history,
                    lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
                )
                if adjusted_answer != full_answer:
                    extra = adjusted_answer[len(full_answer):]
                    if extra:
                        full_answer_parts.append(extra)
                        yield _sse_data(extra.replace("\n", "\\n"))
                    full_answer = adjusted_answer
            if full_answer:
                try:
                    await store.consume_tokens(
                        user_id=user.id,
                        agent_id=request.agent_id,
                        input_tokens=_count_tokens(raw_question),
                        output_tokens=_count_tokens(full_answer),
                        summary_tokens=150,
                    )
                    if is_fallback_response(full_answer):
                        await store.log_fallback(
                            agent_id=request.agent_id,
                            tenant_id=tenant_id,
                            question=raw_question,
                            conversation_id=request.conversation_id,
                        )
                    conversation = await store.append_conversation_messages(
                        agent_id=request.agent_id,
                        tenant_id=tenant_id,
                        user_id=user.id,
                        user_message=raw_question,
                        assistant_message=full_answer,
                        conversation_id=request.conversation_id,
                    )
                    await _maybe_capture_lead(
                        store=store,
                        settings=settings,
                        tenant_id=tenant_id,
                        agent_id=request.agent_id,
                        conversation=conversation,
                        source="chat",
                    )
                    suggestions = await _build_response_suggestions(
                        question,
                        full_answer,
                        agent_doc=agent_doc,
                        intent=intent,
                        conversation_history=conversation_history,
                        cs=cs,
                        agent_id=request.agent_id,
                    )
                    payload = json.dumps({
                        "conversation_id": conversation["id"],
                        "suggestions": suggestions,
                    })
                    yield _sse_event("meta", payload)
                except Exception:
                    pass
            yield _sse_event("done", "[DONE]")

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
    settings = await store.get_settings(request.agent_id, tenant_id)
    raw_question = request.question.strip()
    question  = _strip_greeting_prefix(raw_question) or raw_question

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
        answer = handle_intent(intent, question, agent_doc=agent_doc)
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

    answer = _apply_required_lead_followup(
        answer,
        question=question,
        conversation_history=conversation_history,
        lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
    )

    # Enhancement 8: reliable fallback detection
    if is_fallback_response(answer):
        try:
            await store.log_fallback(
                agent_id=request.agent_id,
                tenant_id=tenant_id,
                question=raw_question,
                conversation_id=request.conversation_id,
            )
        except Exception:
            pass

    conversation = await store.append_conversation_messages(
        agent_id=request.agent_id,
        tenant_id=tenant_id,
        user_id="widget",
        user_message=raw_question,
        assistant_message=answer,
        conversation_id=request.conversation_id,
    )
    try:
        await _maybe_capture_lead(
            store=store,
            settings=settings,
            tenant_id=tenant_id,
            agent_id=request.agent_id,
            conversation=conversation,
            source="widget",
        )
    except Exception:
        pass
    suggestions = await _build_response_suggestions(
        question,
        answer,
        agent_doc=agent_doc,
        intent=intent,
        conversation_history=conversation_history,
        cs=cs,
        agent_id=request.agent_id,
    )
    return ChatResponse(answer=answer, conversation_id=conversation["id"], suggestions=suggestions)


@router.post("/widget/chat/stream")
async def widget_chat_stream(
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
    settings = await store.get_settings(request.agent_id, tenant_id)
    raw_question = request.question.strip()
    question = _strip_greeting_prefix(raw_question) or raw_question

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

    if not last_bot_message and request.last_bot_message and request.last_bot_message.strip():
        last_bot_message = request.last_bot_message.strip()

    if not last_bot_message:
        try:
            recent = await store.db.conversations.find_one(
                {"agent_id": request.agent_id, "tenant_id": tenant_id},
                sort=[("updated_at", -1)],
            )
            if recent:
                recent_msgs = recent.get("messages", [])
                last_bot_message = _get_last_bot_message(recent_msgs)
                if not conversation_history:
                    conversation_history = _get_conversation_history(recent_msgs, max_turns=3)
        except Exception:
            pass

    question = resolve_reference_query(question, last_bot_message)

    intent = detect_intent(question, last_bot_message)
    if intent == "followup_affirmation" and last_bot_message:
        intent = None

    loop = asyncio.get_event_loop()
    pipeline = None if intent else _build_pipeline(store, request.agent_id)
    context = []
    offering_scope = await cs.get_offering_scope_by_agent(request.agent_id)
    if pipeline is not None:
        context = await _build_context(
            pipeline,
            question,
            cs,
            loop,
            agent_id=request.agent_id,
            conversation_history=conversation_history,
        )

    full_answer_parts: list[str] = []

    async def token_stream():
        try:
            if intent:
                answer_text = handle_intent(intent, question, agent_doc=agent_doc)
                full_answer_parts.append(answer_text)
                yield _sse_data(answer_text.replace("\n", "\\n"))
            else:
                gen = await loop.run_in_executor(
                    None,
                    lambda: pipeline.stream_answer_question(
                        question,
                        system_prompt=settings.get("system_prompt"),
                        temperature=settings.get("temperature", 0.2),
                        conversation_history=conversation_history,
                        prefetched_context=context,
                        lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
                        offering_scope=offering_scope,
                    ),
                )
                for token in gen:
                    full_answer_parts.append(token)
                    yield _sse_data(token.replace("\n", "\\n"))
        except Exception as exc:
            yield _sse_event("error", str(exc))
        finally:
            full_answer = "".join(full_answer_parts)
            if full_answer:
                adjusted_answer = _apply_required_lead_followup(
                    full_answer,
                    question=question,
                    conversation_history=conversation_history,
                    lead_capture_enabled=bool(settings.get("lead_capture_enabled")),
                )
                if adjusted_answer != full_answer:
                    extra = adjusted_answer[len(full_answer):]
                    if extra:
                        full_answer_parts.append(extra)
                        yield _sse_data(extra.replace("\n", "\\n"))
                    full_answer = adjusted_answer
            if full_answer:
                try:
                    if is_fallback_response(full_answer):
                        await store.log_fallback(
                            agent_id=request.agent_id,
                            tenant_id=tenant_id,
                            question=raw_question,
                            conversation_id=request.conversation_id,
                        )
                    conversation = await store.append_conversation_messages(
                        agent_id=request.agent_id,
                        tenant_id=tenant_id,
                        user_id="widget",
                        user_message=raw_question,
                        assistant_message=full_answer,
                        conversation_id=request.conversation_id,
                    )
                    await _maybe_capture_lead(
                        store=store,
                        settings=settings,
                        tenant_id=tenant_id,
                        agent_id=request.agent_id,
                        conversation=conversation,
                        source="widget",
                    )
                    suggestions = await _build_response_suggestions(
                        question,
                        full_answer,
                        agent_doc=agent_doc,
                        intent=intent,
                        conversation_history=conversation_history,
                        cs=cs,
                        agent_id=request.agent_id,
                    )
                    payload = json.dumps({
                        "conversation_id": conversation["id"],
                        "suggestions": suggestions,
                    })
                    yield _sse_event("meta", payload)
                except Exception:
                    pass
            yield _sse_event("done", "[DONE]")

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ── Widget settings (public) ──────────────────────────────────────────────────

class PublicSettingsResponse(BaseModel):
    display_name: str
    welcome_message: str
    primary_color: str
    appearance: str
    suggestions: list[str] = Field(default_factory=list)


@router.get("/widget/settings/{agent_id}", response_model=PublicSettingsResponse)
async def widget_settings(
    agent_id: str,
    store: AdminStoreMongo = Depends(_get_store),
    cs: ChunkStore = Depends(_get_chunk_store),
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
        suggestions=await _starter_suggestions_for_agent(agent_doc, cs),
    )
