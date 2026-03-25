from __future__ import annotations
import re


SYSTEM_PROMPT = (
    "You are a friendly customer support assistant for this business.\n"
    "\n"
    "RULE 1 — CONTEXT ONLY (MOST IMPORTANT):\n"
    "Answer ONLY using information explicitly present in the CONTEXT below.\n"
    "NEVER use your own training knowledge to explain, define, or elaborate anything.\n"
    "If the context says \'we offer UI/UX design\' — only say that. "
    "Do NOT explain what UI/UX design is unless the context explains it.\n"
    "If context has no relevant info → say you don\'t have details on that.\n"
    "\n"
    "RULE 2 — NEVER ADD CONTACT SUGGESTIONS WHEN YOU HAVE AN ANSWER:\n"
    "If you answered the question: STOP after the answer and source URL.\n"
    "NEVER add phrases like:\n"
    "  - \'I recommend contacting our team\'\n"
    "  - \'visit our website\'\n"
    "  - \'feel free to reach out\'\n"
    "  - \'for more information contact us\'\n"
    "  - \'you can find more at [website]\'\n"
    "These are FORBIDDEN when you already answered.\n"
    "\n"
    "RULE 3 — WHEN YOU CANNOT FULLY ANSWER:\n"
    "- If you partially answered but there could be more detail, end with:\n"
    "  \'Let me know what you\'d like to know more about!\'\n"
    "- If the context has absolutely no relevant information, say:\n"
    "  \'I don\'t have details on that right now. Let me know if there\'s "
    "something else I can help with!\'\n"
    "  Then show ONLY phone/email if available in context.\n"
    "- NEVER say \'contact our team\' or \'reach out to us\' when you already answered something.\n"
    "\n"
    "RULE 4 — TONE:\n"
    "Be warm and friendly. Start naturally.\n"
    "NEVER end every reply with \'Let me know what you\'d like to know more about!\' "
    "or any fixed closing phrase. This is repetitive and robotic.\n"
    "Vary your endings — sometimes just answer and stop. "
    "Only add a closing when it genuinely fits.\n"
    "\n"
    "RULE 5 — COMPARISONS:\n"
    "List items in plain text, separated by blank lines. No tables, no markdown.\n"
    "\n"
    "RULE 5b — UNCLEAR RESPONSES:\n"
    "If the user\'s reply is vague, ambiguous, or doesn\'t clearly relate to the previous topic "
    "(e.g. just \'yes\', \'ok\', \'sure\', \'and?\', or something unrelated), "
    "do NOT guess or pick a random topic. Instead respond warmly asking for clarification like:\n"
    "  \'I\'m not sure I understood that — could you clarify what you\'d like to know more about?\'\n"
    "  or \'Could you be more specific? I want to make sure I give you the right information!\'\n"
    "Only apply this when the user reply is genuinely unclear in context."
    "\n"
    "RULE 6 — NO SOURCE URLS:\n"
    "Do NOT show or mention any URLs or source links in your response.\n"
    "Never add 🔗 links. Never say \'For more info visit...\'. Just answer."
)

# Comparison trigger words — used to detect comparison questions
COMPARISON_TRIGGERS = [
    "compare", "comparison", "difference", "differences", "vs", "versus",
    "better", "best", "which", "between", "contrast", "different",
]

# Synonyms and paraphrases used for query expansion
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
}


def is_comparison_question(question: str) -> bool:
    """Detect if the question is asking for a comparison."""
    lower = question.lower()
    return any(trigger in lower for trigger in COMPARISON_TRIGGERS)


def expand_query(question: str) -> list[str]:
    """
    Generate 2-3 alternative phrasings using synonym expansion.
    For comparison questions, also adds variants that include all
    relevant comparison terms.
    """
    words = question.lower().split()
    variants = [question]

    # For comparison questions add extra comparison-focused variants
    if is_comparison_question(question):
        # Add a variant that explicitly asks for differences
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

    return variants[:4]  # allow up to 4 for comparison queries


def _format_source_urls(source_urls: list[str] | None) -> str:
    """Source URLs are no longer shown in responses."""
    return ""


def build_messages(
    context: str,
    question: str,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    source_urls: list[str] | None = None,
) -> list[dict[str, str]]:
    # Always merge built-in rules with user's custom prompt.
    # Built-in rules are the strict anti-hallucination / context-only rules.
    # User's custom prompt defines tone, persona, communication style.
    # Built-in rules take priority — user prompt is appended after.
    if system_prompt and system_prompt.strip():
        merged_system = (
            SYSTEM_PROMPT
            + "\n\n---\n\nAdditional instructions for this agent:\n"
            + system_prompt.strip()
        )
    else:
        merged_system = SYSTEM_PROMPT
    """
    Build the message list for the LLM.
    conversation_history: last N messages as [{"role": "user"|"assistant", "content": "..."}]
    Including history lets the LLM understand follow-up questions like "design"
    after a services comparison, instead of treating them as standalone queries.
    """

    # Add comparison-specific instruction if detected
    comparison_hint = ""
    if is_comparison_question(question):
        comparison_hint = (
            "\n\nIMPORTANT: This is a comparison question. "
            "Start with a warm friendly sentence like \"Sure! Here\'s a comparison of...\" or "
            "\"Great question! Let me break that down for you.\" "
            "Then list each item in plain text like:\n"
            "Basic Plan: $10/mo\nFeatures: Chat, Email\n\n"
            "Pro Plan: $25/mo\nFeatures: Chat, Email, API\n\n"
            "End with \"Let me know if you need more details!\" or similar. "
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
        "say you don\'t have that detail and suggest contacting the team. "
        "If you answered from the context: end with \'Let me know what you\'d like to know more about!\' "
        "Never say \'contact our team\' when you have an answer. "
        "Only if context has NO relevant info: say you don\'t have details and show phone/email if in context."
        f"{comparison_hint}"
        f"{_format_source_urls(source_urls)}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
    ]

    # Inject last 4 messages of conversation history so the LLM understands
    # follow-up questions like "design" after a services comparison
    if conversation_history:
        for msg in conversation_history[-4:]:  # last 4 messages (2 turns)
            role = msg.get("role", "")
            text = msg.get("content", "")
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_prompt})
    return messages