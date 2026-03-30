from __future__ import annotations

import os
import re


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a friendly customer support assistant for this business.\n"
    "\n"
    "RULE 1 — CONTEXT ONLY (MOST IMPORTANT):\n"
    "Answer ONLY using information explicitly present in the CONTEXT below.\n"
    "NEVER use your own training knowledge to explain, define, or elaborate anything.\n"
    "If the context says 'we offer UI/UX design' — only say that. "
    "Do NOT explain what UI/UX design is unless the context explains it.\n"
    "If context has no relevant info → say you don't have details on that.\n"
    "\n"
    "RULE 2 — NEVER ADD CONTACT SUGGESTIONS WHEN YOU HAVE AN ANSWER:\n"
    "If you answered the question: STOP after the answer.\n"
    "NEVER add phrases like:\n"
    "  - 'I recommend contacting our team'\n"
    "  - 'visit our website'\n"
    "  - 'feel free to reach out'\n"
    "  - 'for more information contact us'\n"
    "These are FORBIDDEN when you already answered.\n"
    "\n"
    "RULE 3 — WHEN YOU CANNOT FULLY ANSWER:\n"
    "- If you partially answered but there could be more detail, end with a varied closing like:\n"
    "  'Happy to dig deeper on any of these!'\n"
    "  'Which one are you most interested in?'\n"
    "  'Let me know what you'd like to explore further.'\n"
    "- If the context has absolutely no relevant information, say:\n"
    "  'I don't have details on that right now. Let me know if there's something else I can help with!'\n"
    "  Then show ONLY phone/email if available in context.\n"
    "- NEVER say 'contact our team' or 'reach out to us' when you already answered something.\n"
    "\n"
    "RULE 4 — TONE AND HUMAN TOUCH:\n"
    "Be warm, conversational, and genuinely helpful — like a knowledgeable friend, not a FAQ page.\n"
    "\n"
    "When presenting a list of products or services:\n"
    "  - Open with a natural sentence, e.g. 'Here's what we offer:' or 'You've got quite a few options!'\n"
    "  - After the list, add ONE short human sentence that invites the next step.\n"
    "    Good: 'Happy to dig into any of these — just say the word!'\n"
    "    Good: 'Which one sounds most relevant to what you're working on?'\n"
    "    Good: 'Any of these catch your eye?'\n"
    "  - Vary that closing every time — never repeat the same phrase twice in a row.\n"
    "\n"
    "Vary how you open answers too. Instead of always starting the same way, try:\n"
    "  'Great question!' / 'Sure!' / 'Absolutely!' / 'Here's the full picture:'\n"
    "\n"
    "NEVER end every reply with 'Let me know what you'd like to know more about!' "
    "or any fixed closing phrase. This is repetitive and robotic.\n"
    "Vary your endings — sometimes just answer and stop. "
    "Only add a closing when it genuinely fits.\n"
    "\n"
    "RULE 5 — COMPARISONS:\n"
    "List items in plain text, separated by blank lines. No tables, no markdown.\n"
    "\n"
    "RULE 5b — UNCLEAR RESPONSES:\n"
    "If the user's reply is vague, ambiguous, or doesn't clearly relate to the previous topic "
    "(e.g. just 'yes', 'ok', 'sure', 'and?', or something unrelated), "
    "do NOT guess or pick a random topic. Instead respond warmly asking for clarification like:\n"
    "  'I'm not sure I understood that — could you clarify what you'd like to know more about?'\n"
    "  or 'Could you be more specific? I want to make sure I give you the right information!'\n"
    "Only apply this when the user reply is genuinely unclear in context.\n"
    "\n"
    "RULE 5c — SINGLE PRODUCT FOCUS:\n"
    "When the question is specifically about ONE named product (e.g. 'Tell me more about ERP-Integrated CRM'), "
    "answer ONLY about that product. "
    "Do NOT list all products again. "
    "Do NOT mention other products unless directly comparing. "
    "Go deep on the one product: what it does, key features, integrations, benefits.\n"
    "\n"
    "RULE 6 — NO SOURCE URLS:\n"
    "Do NOT show or mention any URLs or source links in your response.\n"
    "Never add \U0001f517 links. Never say 'For more info visit...'. Just answer.\n"
    "\n"
    "RULE 7 — RESOLVE REFERENCES FROM HISTORY:\n"
    "When the user says 'the first one', 'the second one', 'that product', 'it', etc.,\n"
    "look at the CONVERSATION HISTORY to identify exactly which item they mean.\n"
    "Map ordinals (first=1, second=2, third=3...) to the numbered list in the previous bot reply.\n"
    "Then answer about THAT specific item using the CONTEXT.\n"
    "NEVER guess or answer about a different item. If you cannot resolve, ask for clarification."
)

# ── Comparison / list detection ────────────────────────────────────────────────

COMPARISON_TRIGGERS = [
    "compare", "comparison", "difference", "differences", "vs", "versus",
    "better", "best", "which", "between", "contrast", "different",
]

LIST_TRIGGERS = [
    "all", "list", "every", "what do you offer", "what products",
    "what services", "what do you have", "show me all", "full list",
    "how many products", "how many services",
]

QUERY_SYNONYMS: dict[str, list[str]] = {
    "offer":      ["provide", "service", "sell", "have", "offer"],
    "provide":    ["offer", "service", "sell", "have", "provide"],
    "services":   ["offerings", "products", "solutions", "services", "packages"],
    "products":   ["services", "offerings", "items", "solutions", "products"],
    "price":      ["cost", "fee", "charge", "rate", "pricing", "price"],
    "pricing":    ["cost", "fee", "charge", "rate", "price", "pricing"],
    "cost":       ["price", "fee", "charge", "rate", "pricing", "cost"],
    "plan":       ["package", "tier", "subscription", "option", "plan"],
    "plans":      ["packages", "tiers", "subscriptions", "options", "plans"],
    "package":    ["plan", "tier", "subscription", "option", "package"],
    "compare":    ["difference", "vs", "versus", "between", "compare"],
    "difference": ["compare", "vs", "versus", "between", "difference"],
    "buy":        ["purchase", "order", "get", "acquire", "buy"],
    "purchase":   ["buy", "order", "get", "acquire", "purchase"],
    "contact":    ["reach", "call", "email", "get in touch", "contact"],
    "reach":      ["contact", "call", "email", "get in touch", "reach"],
    "location":   ["address", "where", "place", "office", "location"],
    "address":    ["location", "where", "place", "office", "address"],
    "hours":      ["time", "schedule", "open", "available", "hours"],
    "available":  ["offer", "have", "provide", "sell", "available"],
    "help":       ["assist", "support", "service", "help"],
    "support":    ["help", "assist", "service", "support"],
    "team":       ["staff", "people", "employees", "who", "team"],
    "about":      ["who", "company", "history", "background", "about"],
    "work":       ["services", "do", "offer", "provide", "work"],
    "do":         ["offer", "provide", "service", "work", "do"],
    "feature":    ["benefit", "include", "capability", "function", "feature"],
    "features":   ["benefits", "includes", "capabilities", "functions", "features"],
    "benefit":    ["feature", "advantage", "include", "perk", "benefit"],
    "integrate":  ["connect", "sync", "link", "work with", "integrate"],
    "integration":["connector", "connection", "plugin", "sync", "integration"],
}


def is_comparison_question(question: str) -> bool:
    lower = question.lower()
    return any(trigger in lower for trigger in COMPARISON_TRIGGERS)


def is_list_question(question: str) -> bool:
    lower = question.lower()
    return any(trigger in lower for trigger in LIST_TRIGGERS)


# ── Enhancement 5: LLM-assisted query rewriting ────────────────────────────────

def rewrite_query_with_llm(question: str, llm_client) -> list[str]:
    """
    Use gpt-4o-mini to generate 2 semantically distinct phrasings of the query.
    Falls back silently to an empty list if the call fails.
    Only called when synonym expansion produces fewer than 2 non-trivial variants.
    """
    if llm_client is None:
        return []
    try:
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite search queries. "
                        "Given a question, output exactly 2 alternative phrasings "
                        "that mean the same thing but use different words. "
                        "Output only the 2 phrasings, one per line, no numbering, no extra text."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=120,
            temperature=0.4,
        )
        raw = (resp.choices[0].message.content or "").strip()
        rewrites = [line.strip() for line in raw.splitlines() if line.strip()]
        return rewrites[:2]
    except Exception:
        return []


def expand_query(question: str, llm_client=None) -> list[str]:
    """
    Generate up to 4 alternative phrasings:
      1. Original question (always first)
      2. Comparison-specific variants (if detected)
      3. Synonym-expansion variant
      4. LLM-rewritten variants (if llm_client provided and synonyms gave < 2 extras)
    """
    words = question.lower().split()
    variants: list[str] = [question]

    if is_comparison_question(question):
        if "compare" not in question.lower():
            variants.append("compare " + question.lower())
        if "difference" not in question.lower():
            variants.append("difference between " + question.lower())

    # Synonym expansion
    for i, word in enumerate(words):
        clean_word = word.strip("?.,!")
        if clean_word in QUERY_SYNONYMS:
            for synonym in QUERY_SYNONYMS[clean_word][:2]:
                new_words = words.copy()
                new_words[i] = synonym
                variant = " ".join(new_words).strip()
                if variant != question.lower() and variant not in [v.lower() for v in variants]:
                    variants.append(variant)
            break

    # LLM rewrite when synonyms didn't give enough diversity
    if llm_client is not None and len(variants) < 3:
        llm_variants = rewrite_query_with_llm(question, llm_client)
        for v in llm_variants:
            if v.lower() not in [x.lower() for x in variants]:
                variants.append(v)

    return variants[:4]


# ── Message builder ────────────────────────────────────────────────────────────

def _format_source_urls(source_urls: list[str] | None) -> str:
    """Source URLs are not shown in responses per RULE 6."""
    return ""


def build_messages(
    context: str,
    question: str,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    source_urls: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build the message list for the LLM.

    The built-in SYSTEM_PROMPT (anti-hallucination + tone rules) is always
    merged with the agent's custom system_prompt so persona settings are
    honoured without bypassing the safety rules.

    conversation_history: last N messages as [{"role": "user"|"assistant", "content": "..."}]
    Including history lets the LLM resolve follow-up questions like "tell me more
    about #3" or "which one is cheaper?" correctly.
    """
    # Enhancement (previous session): always use merged_system so custom agent
    # prompts are actually injected instead of silently discarded.
    if system_prompt and system_prompt.strip():
        merged_system = (
            SYSTEM_PROMPT
            + "\n\n---\n\nAdditional instructions for this agent:\n"
            + system_prompt.strip()
        )
    else:
        merged_system = SYSTEM_PROMPT

    comparison_hint = ""
    if is_comparison_question(question):
        comparison_hint = (
            "\n\nIMPORTANT: This is a comparison question. "
            "Start with a warm friendly sentence like \"Sure! Here's a comparison of...\" or "
            "\"Great question! Let me break that down for you.\" "
            "Then list each item in plain text like:\n"
            "Basic Plan: $10/mo\nFeatures: Chat, Email\n\n"
            "Pro Plan: $25/mo\nFeatures: Chat, Email, API\n\n"
            "End with a varied human closing. "
            "Do NOT use tables or markdown. No robotic tone."
        )

    user_prompt = (
        "CONTEXT (answer ONLY from this, nothing else):\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "IMPORTANT: Use ONLY the information in the CONTEXT above. "
        "Do not add general knowledge or explanations. "
        "If the context does not contain enough information to answer fully, "
        "say you don't have that detail and suggest contacting the team. "
        "Never say 'contact our team' when you already have an answer. "
        "Only if context has NO relevant info: say you don't have details and show phone/email if in context."
        f"{comparison_hint}"
        f"{_format_source_urls(source_urls)}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": merged_system},
    ]

    # Enhancement 1: inject full conversation history (not just the last bot message)
    # so follow-ups like "tell me more about #3" or "which is cheaper?" resolve correctly.
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 3 turns (6 messages)
            role = msg.get("role", "")
            text = msg.get("content", "")
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_prompt})
    return messages