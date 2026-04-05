from __future__ import annotations

import os
import re

_CONTACT_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_CONTACT_PHONE_RE = re.compile(r"(?:\+\d[\d\s().-]{7,}\d|\b\d{7,}\b)")
_CONTACT_PLACEHOLDER_RE = re.compile(
    r"(?i)\b(?:email|mail|e-mail|phone|mobile|contact)\s*:\s*(?:\[\s*not available\s*\]|not available|n/?a|na|unknown|-)\b"
)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a friendly customer support assistant for this business.\n"
    "\n"
    "RULE 1 — CONTEXT ONLY (MOST IMPORTANT):\n"
    "Answer ONLY using information explicitly present in the CONTEXT below.\n"
    "NEVER use your own training knowledge to explain, define, or elaborate anything.\n"
    "If the context says 'we offer UI/UX design' — only say that. "
    "Do NOT explain what UI/UX design is unless the context explains it.\n"
    "If context has no relevant info → respond warmly and positively without guessing.\n"
    "\n"
    "RULE 1b — NEVER OMIT ITEMS WHEN LISTING:\n"
    "When listing products or services from context, you MUST include EVERY item present.\n"
    "Do NOT skip, merge, or omit any item — even if it looks different from the others.\n"
    "Count the items in context first, then make sure your list has the exact same count.\n"
    "Example: if context has 8 Product:/Service: lines, your list must have exactly 8 items.\n"
    "When you present a list, format it as a numbered list using '1.', '2.', '3.' and put each item on its own line.\n"
    "For 'what do you offer' or 'offerings' questions, include both Product and Service items when both exist in context.\n"
    "If the context contains only Product items or only Service items, list only those as the offering.\n"
    "\n"
    "RULE 2 — NEVER ADD CONTACT SUGGESTIONS WHEN YOU HAVE AN ANSWER:\n"
    "If you answered the question: STOP after the answer.\n"
    "NEVER add phrases like:\n"
    "  - 'I recommend contacting our team'\n"
    "  - 'visit our website'\n"
    "  - 'feel free to reach out'\n"
    "  - 'for more information contact us'\n"
    "These are FORBIDDEN when you already answered.\n"
    "Exception: if the question is specifically about pricing and the context does not contain pricing details, "
    "you may politely direct the user to the sales team and include both the sales phone number and sales email only if those exact details are present in the context.\n"
    "Exception: if you are recommending a specific product, solution, or service as the best fit for the user's needs, "
    "you may add one short professional sentence saying the sales team can help discuss the requirement in more detail and provide a tailored solution.\n"
    "If contact details are available in the context for that recommendation handoff, include both the phone number and email address. "
    "If only one is available, include only that one.\n"
    "When you include contact details, put them in the next paragraph on separate labeled lines so they stand out clearly.\n"
    "Never output placeholder examples or made-up contact details. If no real contact details are present in the context, omit them entirely.\n"
    "Never write placeholder contact values such as '[not available]', 'not available', 'N/A', 'NA', 'unknown', or '-'.\n"
    "\n"
    "RULE 3 — WHEN YOU CANNOT FULLY ANSWER:\n"
    "- If you partially answered but there could be more detail, end with a varied closing like:\n"
    "  'Happy to dig deeper on any of these!'\n"
    "  'Which one are you most interested in?'\n"
    "  'Let me know what you'd like to explore further.'\n"
    "- If the context has absolutely no relevant information, reply in a professional, diplomatic, and polite way.\n"
    "  Example tone: 'Thank you for your question. At the moment, I do not have the relevant information available. If you would like, please let me know if there is anything else I can help you with.'\n"
    "  Then show phone/email only if available in context.\n"
    "- If the question is about price, pricing, cost, quote, fees, or charges and the context does not contain that information,\n"
    "  respond politely and professionally that pricing details are not available in the current information,\n"
    "  and say the sales team can help. Then include both the sales phone number and the sales email if both are explicitly available in the context.\n"
    "  If only one of them is available, include whichever is present.\n"
    "  Put those contact details in the next paragraph, not inside the explanatory paragraph.\n"
    "  Show each one on its own labeled line with clear labels such as Email and Phone.\n"
    "  If no real contact details are present in the context, do not output any contact block.\n"
    "  Never output placeholder contact values such as '[not available]', 'not available', 'N/A', 'NA', 'unknown', or '-'.\n"
    "  After sharing the contact details, end with one short professional follow-up invitation such as:\n"
    "  'If you'd like, I can also help with anything else you need.'\n"
    "  or 'Please let me know if there's anything else you'd like to know.'\n"
    "  Example tone: 'Thank you for your interest in pricing. At the moment, I don't have access to specific pricing details. "
    "Since pricing can vary depending on your exact requirements, our sales team would be best positioned to provide accurate and tailored information. "
    "I'd be happy to connect you with them to discuss your needs further.'\n"
    "- NEVER say 'contact our team' or 'reach out to us' when you already answered something.\n"
    "\n"
    "RULE 4 — TONE AND HUMAN TOUCH:\n"
    "Be warm, conversational, and genuinely helpful — like a knowledgeable friend, not a FAQ page.\n"
    "Avoid stiff or robotic openings such as 'Here is the offer', 'Here are the offerings', or repetitive template phrasing.\n"
    "Start in a more human way, like you're naturally replying to a person.\n"
    "For any answer, prefer a warm, polished, professional opening when it fits naturally.\n"
    "Good openings: 'Certainly.' / 'Of course.' / 'Sure — here's an overview.' / 'Here's a clear overview.'\n"
    "Avoid cold or mechanical openings like: 'Answer:' / 'Here is the information:' / 'Based on the context provided:'\n"
    "Do not begin a normal answered reply with 'Thank you for your question' when the answer is available.\n"
    "Reserve that style for fallback replies or pricing handoff cases where the requested information is unavailable.\n"
    "Do not say things like 'in the context provided', 'based on the provided context', or 'according to the context above' in the final reply.\n"
    "Simply answer the user's question directly and naturally.\n"
    "For any answer, close in a professional, natural, and diplomatic way too.\n"
    "If a closing fits, keep it short, polished, and helpful.\n"
    "Good closings: 'I would be glad to provide more detail if that would be helpful.' / 'Please let me know if you would like a more detailed overview of any part.' / 'I would be happy to provide further information if needed.'\n"
    "Avoid repetitive closings and avoid sounding scripted.\n"
    "Human-style examples by question type:\n"
    "  - Product question: 'Certainly — here's a brief overview of that product.'\n"
    "  - Service question: 'Of course — here's how that service is described.'\n"
    "  - Pricing question: 'Certainly — here's what I can see regarding pricing.'\n"
    "  - General business question: 'Certainly.'\n"
    "  - Fallback: 'Thank you for your question. At the moment, I do not have the relevant information available.'\n"
    "Human-style closings by question type:\n"
    "  - Product/service answer: 'I would be happy to provide more detail on any part if helpful.'\n"
    "  - Pricing answer: 'I would be glad to help further if you would like to explore the available options.'\n"
    "  - General answer: 'I hope this provides a helpful overview.'\n"
    "  - Recommendation answer: 'For a more detailed discussion around your specific requirements, our sales team would be pleased to help and can provide a tailored solution.'\n"
    "    If contact details are available, include them clearly in the next paragraph on separate labeled lines.\n"
    "  - Fallback: 'Please let me know if there is anything else I may assist you with.'\n"
    "For recommendation questions, only recommend a specific product, solution, or service when the user's requirements are already clear from the current question or the earlier conversation.\n"
    "If the requirement is not yet clear, do not guess. Ask a short, polite clarification question first.\n"
    "\n"
    "When presenting a list of products or services:\n"
    "  - Open with a professional human sentence, not a dry label.\n"
    "    Good: 'Certainly — here's an overview of what we offer.'\n"
    "    Good: 'Of course — here is a summary of our offerings.'\n"
    "    Good: 'Sure — we offer a range of products and services.'\n"
    "    Avoid: 'Here is the offer.' / 'Here are the offerings.'\n"
    "  - Use numbered lists, one item per line.\n"
    "    Good:\n"
    "    1. ERP Integrated CRM\n"
    "    2. ERP-Integrated Field Service Management\n"
    "    3. ERP-Integrated Warehouse & Inventory Management\n"
    "  - Do not run multiple list items together in one paragraph.\n"
    "  - After the list, add ONE short professional sentence that invites the next step.\n"
    "    Good: 'I would be happy to provide more detail on any of these if helpful.'\n"
    "    Good: 'Please let me know if you would like a closer look at any of these options.'\n"
    "    Good: 'I would be glad to explain any of these in more detail if needed.'\n"
    "  - Vary that closing every time — never repeat the same phrase twice in a row.\n"
    "\n"
    "Vary how you open answers too. Instead of always starting the same way, try:\n"
    "  'Certainly.' / 'Of course.' / 'Sure.' / 'Here's an overview:'\n"
    "For offer/product/service questions, prefer natural openers like:\n"
    "  'Certainly — here's an overview of what we offer.'\n"
    "  'Of course — here is a summary of our offerings.'\n"
    "  'We offer a mix of products and services.'\n"
    "For non-list questions, use the same idea: answer naturally, then end with a brief professional closing only if it adds value.\n"
    "Avoid casual closings such as 'feel free to ask' or 'If you have any other questions or need further information, feel free to ask!'\n"
    "Prefer more diplomatic phrasing such as 'Please let me know if you would like any additional information.' or 'I would be happy to assist further if needed.'\n"
    "\n"
    "NEVER end every reply with 'Let me know what you'd like to know more about!' "
    "or any fixed closing phrase. This is repetitive and robotic.\n"
    "Vary your endings — sometimes just answer and stop. "
    "Only add a closing when it genuinely fits.\n"
    "\n"
    "RULE 5 — COMPARISONS:\n"
    "For general lists, present items line by line using numbered lists. No tables for plain lists.\n"
    "For comparison questions (when the user asks to compare two or more items), follow the COMPARISON ENGINE instructions injected in the user prompt exactly. "
    "Tables ARE allowed and expected inside structured comparison responses.\n"
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
    "RULE 6b — NEVER INVENT CONTACT DETAILS:\n"
    "Only mention a phone number or email address if it is explicitly present in the CONTEXT.\n"
    "If a phone number is not present in the CONTEXT, do NOT create, guess, infer, or format one.\n"
    "If an email address is not present in the CONTEXT, do NOT create, guess, or infer one.\n"
    "If only one contact detail is present, mention only that one.\n"
    "If no contact detail is present, omit the contact section entirely.\n"
    "Never output placeholder contact values such as '[not available]', 'not available', 'N/A', 'NA', 'unknown', or '-'.\n"
    "Never fabricate placeholder-style contact details.\n"
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
    # Explicit listing phrases
    "all", "list", "every", "full list",
    # Direct offer / product queries
    "what do you offer", "what products", "what services",
    "what do you have", "what do you provide",
    "show me all", "show me your", "show me what",
    "how many products", "how many services",
    # Natural conversational variants (previously missed)
    "what can you do", "what solutions", "what options",
    "what are your", "what are the products", "what are the services",
    "tell me about your products", "tell me about your services",
    "tell me what you offer", "tell me your products",
    "do you have any products", "do you have any services",
    "which products", "which services",
    "give me a list", "give me all",
    "overview of your", "summary of your",
]

QUERY_SYNONYMS: dict[str, list[str]] = {
    "offer":      ["provide", "sell", "have", "offer", "deliver"],
    "provide":    ["offer", "deliver", "sell", "have", "provide"],
    "services":   ["offerings", "services", "packages", "support", "consulting"],
    "products":   ["offerings", "items", "solutions", "products", "platforms"],
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


def extract_comparison_items(question: str) -> list[str]:
    """
    Extract individual items being compared from a comparison question.

    Examples:
      "compare Automotive Greases and Extreme Pressure Greases"
        → ["Automotive Greases", "Extreme Pressure Greases"]
      "Grease A vs Grease B"
        → ["Grease A", "Grease B"]
    Returns an empty list if no clear items can be extracted.
    """
    # Strip leading comparison trigger words to isolate the subject phrase
    cleaned = re.sub(
        r"(?i)^(compare|comparison of|difference between|differences between|contrast)\s+",
        "",
        question.strip(),
    ).strip()

    # Try each split pattern in priority order
    for pattern in (
        r"\s+vs\.?\s+",
        r"\s+versus\s+",
        r"\s+and\s+",
        r"\s+or\s+",
        r"\s+between\s+",
    ):
        parts = re.split(pattern, cleaned, flags=re.I)
        if len(parts) >= 2:
            items = [p.strip().strip("?.,!:") for p in parts if p.strip()]
            # Only return if every item looks like a real noun phrase (> 2 chars)
            if all(len(item) > 2 for item in items):
                return items

    return []


def is_pricing_question(question: str) -> bool:
    lower = question.lower()
    return bool(re.search(
        r"\b(price|pricing|cost|costs|quote|quotes|fee|fees|charge|charges|"
        r"how much|subscription|plan|plans|rate|rates)\b",
        lower,
    ))


def is_career_question(question: str) -> bool:
    lower = (question or "").lower()
    return bool(re.search(
        r"\b(job|jobs|career|careers|opening|openings|open position|open positions|"
        r"vacancy|vacancies|hiring|apply|application|recruitment|cv|resume)\b",
        lower,
    ))


def is_recommendation_question(question: str) -> bool:
    lower = question.lower()
    return bool(re.search(
        r"\b(recommend|suggest|best|better|suitable|suit|right fit|fit for|"
        r"which.*(product|service|solution|option)|"
        r"what.*(product|service|solution).*(for me|for us|for our)|"
        r"which one should i choose|which should i choose|what should i choose)\b",
        lower,
    ))


_RECOMMENDATION_REQUIREMENT_HINTS = (
    "need", "needs", "require", "requires", "requirement", "requirements",
    "looking for", "looking to", "want", "wants", "for our", "for us", "for me",
    "sales", "crm", "warehouse", "inventory", "manufacturing", "shopfloor",
    "field service", "service team", "partner portal", "portal", "configurator",
    "asset", "tracking", "integration", "integrate", "sap", "oracle", "epicor",
    "support", "implementation", "migration", "training", "security", "compliance",
    "offline", "mobile", "iot", "rfid", "barcode", "automation", "ai",
)


def _text_has_recommendation_requirements(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False

    if any(hint in lower for hint in _RECOMMENDATION_REQUIREMENT_HINTS):
        return True

    words = re.findall(r"[a-z0-9]+", lower)
    if len(words) >= 7 and any(token in lower for token in ("for", "need", "want", "looking")):
        return True

    return False


def has_recommendation_requirements(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    if _text_has_recommendation_requirements(question):
        return True

    if not conversation_history:
        return False

    for msg in reversed(conversation_history):
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content") or "").strip()
        if _text_has_recommendation_requirements(text):
            return True

    return False


def detect_pricing_subject(question: str) -> str | None:
    list_category = detect_list_category(question)
    if list_category == "product":
        return "products"
    if list_category == "service":
        return "services"
    if list_category == "offering":
        return "products and services"

    lower = question.lower()
    if re.search(r"\bproduct|products\b", lower):
        return "products"
    if re.search(r"\bservice|services\b", lower):
        return "services"
    if re.search(r"\boffer|offering|offerings\b", lower):
        return "products and services"
    return None


def is_list_question(question: str) -> bool:
    lower = question.lower()
    if any(trigger in lower for trigger in LIST_TRIGGERS):
        return True

    # Handle looser / ungrammatical variants such as:
    # "what is upbuff products", "tell upbuff services", "upbuff products"
    # These should still route to the exhaustive list-answer path.
    has_catalog_term = bool(re.search(
        r"\b(product|products|service|services|solution|solutions|offer|offers|offering|offerings|provide|provides|provided)\b",
        lower,
    ))
    has_list_intent = bool(re.search(
        r"\b(what|which|list|show|tell|give|all|available|offer|offers|offered|"
        r"have|has|provide|provides|overview|summary)\b",
        lower,
    ))

    if has_catalog_term and has_list_intent:
        return True

    # Short business-catalog phrases like "your offer" or "your offerings"
    # should route to list retrieval even without an explicit verb such as "show".
    if has_catalog_term and re.search(r"\b(your|you)\b", lower):
        return True

    # Very short noun-phrase queries like "upbuff products" should also count.
    compact = lower.strip(" ?!.")
    if has_catalog_term and len(compact.split()) <= 4:
        return True

    return False


def detect_list_category(question: str) -> str | None:
    """
    Detect whether a list-style question targets products, services, or the
    combined set of business offerings.

    Returns:
      - "product" when the wording clearly targets products/solutions
      - "service" when the wording clearly targets services
      - "offering" when the wording asks what the business offers/provides or
        mixes products and services together
      - None for generic list wording that doesn't imply a product/service catalog
    """
    lower = question.lower()

    product_terms = (
        " product", " products", "solution", "solutions", "item", "items",
        "catalog", "catalogue", "platform", "platforms",
    )
    service_terms = (
        " service", " services", "consulting",
        "support", "maintenance", "implementation", "migration", "training",
    )
    offering_terms = (
        " offer", " offers", "offering", "offerings", "provide", "provides",
        "provided", "what do you do", "what can you do",
    )

    product_hit = any(term in f" {lower}" for term in product_terms)
    service_hit = any(term in f" {lower}" for term in service_terms)
    offering_hit = any(term in f" {lower}" for term in offering_terms)

    if product_hit and service_hit:
        return "offering"
    if product_hit and not service_hit:
        return "product"
    if service_hit and not product_hit:
        return "service"
    if offering_hit:
        return "offering"
    return None


def detect_context_offering_scope(context: str) -> str | None:
    has_products = bool(re.search(r"(?im)^\s*Product\s*:", context or ""))
    has_services = bool(re.search(r"(?im)^\s*Service\s*:", context or ""))
    if has_products and has_services:
        return "offering"
    if has_products:
        return "product"
    if has_services:
        return "service"
    return None


def is_qualified_inquiry_question(question: str) -> bool:
    lower = question.lower()
    return bool(re.search(
        r"\b("
        r"demo|book a demo|schedule a demo|request a demo|"
        r"quote|quotation|pricing|price|cost|proposal|rfq|"
        r"service request|support request|implementation request|"
        r"customization|customisation|customize|customise|"
        r"integration|integrations|integrate|"
        r"inspection|assessment|audit|site visit|"
        r"erp customization|erp customisation|erp inspection|erp assessment|erp audit"
        r")\b",
        lower,
    ))


def is_demo_request_question(question: str) -> bool:
    lower = question.lower()
    return bool(re.search(r"\b(demo|book a demo|schedule a demo|request a demo)\b", lower))


def is_appointment_request_question(question: str) -> bool:
    lower = question.lower()
    return bool(re.search(r"\b(appointment|book an appointment|schedule an appointment)\b", lower))


def is_lead_capture_intent_question(question: str) -> bool:
    lower = (question or "").lower().strip()
    if not lower:
        return False

    if (
        is_demo_request_question(lower)
        or is_appointment_request_question(lower)
        or is_pricing_question(lower)
        or is_qualified_inquiry_question(lower)
    ):
        return True

    return bool(re.search(
        r"\b("
        r"interested|interested in|showing interest|"
        r"looking to buy|ready to buy|want to buy|purchase|buy|"
        r"contact me|call me|reach me|reach out|follow up|callback|"
        r"consultation|let'?s talk|talk to someone|get in touch|"
        r"custom solution|tailored solution"
        r")\b",
        lower,
    ))


def has_qualified_inquiry(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    if is_qualified_inquiry_question(question):
        return True

    if not conversation_history:
        return False

    for msg in reversed(conversation_history):
        if msg.get("role") != "user":
            continue
        if is_qualified_inquiry_question(str(msg.get("content") or "")):
            return True

    return False


_CONTACT_DETAIL_RE = re.compile(
    r"("
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"(?:\+\d[\d\s().-]{7,}\d|\b\d{6,15}\b)"
    r")",
    re.I,
)


def _text_has_contact_details(text: str) -> bool:
    return bool(_CONTACT_DETAIL_RE.search(text or ""))


def _is_compact_customer_contact_submission(text: str) -> bool:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return False

    email_present = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", normalized, re.I))
    digit_groups = re.findall(r"\b\d[\d\s().+-]{4,}\d\b|\b\d{6,15}\b", normalized)
    word_count = len(re.findall(r"[A-Za-z0-9@._%+-]+", normalized))
    return word_count <= 8 and (email_present or bool(digit_groups))


def _extract_customer_email(text: str) -> str | None:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "", re.I)
    return match.group(0).lower() if match else None


def _extract_customer_phone(text: str) -> str | None:
    match = re.search(r"(?:\+\d[\d\s().-]{7,}\d|\b\d{10,}\b)", text or "")
    if match:
        return match.group(0).strip()

    if _is_compact_customer_contact_submission(text):
        compact_match = re.search(r"\b\d{6,15}\b", text or "")
        if compact_match:
            return compact_match.group(0).strip()
    return None


def _extract_customer_name(text: str) -> str | None:
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
        candidate = re.split(
            r"(?i)\b(?:my email is|email is|you can reach me|reach me at|phone is|mobile is|contact me at|from)\b",
            match.group(1),
            maxsplit=1,
        )[0]
        candidate = re.split(r"[,.!;:\n]", candidate, maxsplit=1)[0].strip()
        if candidate and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", candidate):
            words = [word for word in candidate.split() if word]
            if len(words) <= 4:
                return candidate

    if _is_compact_customer_contact_submission(text):
        stripped = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", text or "", flags=re.I)
        stripped = re.sub(r"(?:\+\d[\d\s().-]{7,}\d|\b\d{6,15}\b)", " ", stripped)
        stripped = re.sub(r"[^A-Za-z .'-]", " ", stripped)
        stripped = " ".join(stripped.split())
        if stripped and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", stripped):
            words = [word for word in stripped.split() if word]
            if len(words) <= 4:
                return stripped
    return None


def _extract_customer_company(text: str) -> str | None:
    patterns = (
        r"\b(?:company|organization|organisation|business)\s*(?:name)?\s*(?:is|:)\s*([A-Za-z0-9][A-Za-z0-9 &.,'-]{1,80})",
        r"\b(?:i am|i'm)\s+from\s+([A-Za-z0-9][A-Za-z0-9 &.,'-]{1,80})",
        r"\b(?:we are|we're)\s+from\s+([A-Za-z0-9][A-Za-z0-9 &.,'-]{1,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if match:
            candidate = match.group(1).strip(" ,.;:-")
            if candidate:
                return candidate
    return None


def _text_has_interest_details(text: str) -> bool:
    lower = (text or "").lower().strip()
    if not lower:
        return False
    if is_qualified_inquiry_question(lower) and len(re.findall(r"[a-z0-9]+", lower)) >= 5:
        return True
    return bool(re.search(
        r"\b(need|needs|looking for|looking to|want|wants|require|requires|use case|for our|for us|for me|interested in|specific requirement|unique requirement)\b",
        lower,
    )) and len(re.findall(r"[a-z0-9]+", lower)) >= 5


def _extract_demo_target_candidate(text: str) -> str | None:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return None

    generic_targets = {
        "a product",
        "the product",
        "product",
        "products",
        "a service",
        "the service",
        "service",
        "services",
        "a solution",
        "the solution",
        "solution",
        "solutions",
        "your product",
        "your products",
        "your service",
        "your services",
        "your solution",
        "your solutions",
        "more information",
        "pricing",
        "a demo",
        "the demo",
        "demo",
        "an appointment",
        "the appointment",
        "appointment",
    }

    def clean_candidate(candidate: str | None) -> str | None:
        if not candidate:
            return None
        cleaned = candidate.strip(" .,:;!?-")
        cleaned = re.sub(
            r"^(?:(?:i am|i'm)\s+interested in|interested in|looking for|looking to|want to|want|need to|need|book|schedule|request)\s+",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(
            r"^(?:(?:i\s+)?(?:need|want|would\s+like|would\s+love|am\s+looking)\s+to\s+)?"
            r"(?:book|schedule|request)\s+",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.I)
        cleaned = cleaned.strip(" .,:;!?-")
        if re.fullmatch(
            r"(?:(?:i\s+)?(?:need|want|would\s+like|would\s+love|am\s+looking)\s+to\s+)?"
            r"(?:book|schedule|request)?\s*(?:a|an|the)?",
            cleaned,
            re.I,
        ):
            return None
        if not cleaned or cleaned.lower() in generic_targets:
            return None
        return cleaned

    referential_match = re.search(
        r"\b(?:this|that|the first|the second|the third|first|second|third)\s+(?:product|service|solution|offering)\b",
        normalized,
        re.I,
    )
    if referential_match:
        return referential_match.group(0)

    patterns = (
        r"\b(?:demo|appointment|meeting|call)\b.{0,40}?\b(?:for|about|on)\s+([A-Za-z0-9][A-Za-z0-9 &+/\-]{1,60})",
        r"\b([A-Za-z0-9][A-Za-z0-9 &+/\-]{1,60})\s+(?:demo|appointment|meeting|call)\b",
    )
    for pattern in patterns:
        target_match = re.search(pattern, normalized, re.I)
        candidate = clean_candidate(target_match.group(1) if target_match else None)
        if candidate:
            return candidate
    return None


def has_demo_target_context(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    texts = [question]
    if conversation_history:
        texts.extend(
            str(msg.get("content") or "")
            for msg in conversation_history
            if msg.get("role") == "user"
        )

    for text in texts:
        if _extract_demo_target_candidate(text):
            return True
    return False


def should_clarify_demo_target(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    if not (is_demo_request_question(question) or is_appointment_request_question(question)):
        return False
    return not has_demo_target_context(question, conversation_history)


def _is_information_request_question(text: str) -> bool:
    lower = (text or "").lower().strip()
    if not lower:
        return False
    return bool(re.search(
        r"\b("
        r"tell me more|know more|what is|what are|which|how does|how do|how can|"
        r"price|pricing|cost|products?|services?|features?|benefits?|details?|"
        r"included|difference|differ|compare|comparison|overview|fit for us|"
        r"product|service|solution|offerings?"
        r")\b",
        lower,
    ))


def _is_affirmative_demo_followup(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    normalized = re.sub(r"[!?.]+$", "", (question or "").strip().lower()).strip()
    if normalized not in {
        "yes",
        "yes please",
        "yep",
        "yeah",
        "yup",
        "sure",
        "sure please",
        "ok",
        "okay",
        "go ahead",
        "please do",
    }:
        return False

    if not conversation_history:
        return False

    recent_assistant_demo_offer = False
    for msg in reversed(conversation_history[-6:]):
        if msg.get("role") != "assistant":
            continue
        lower = str(msg.get("content") or "").lower()
        if (
            "would you like to proceed with booking" in lower
            or "would you like to proceed with scheduling" in lower
            or "would you like to book that demo" in lower
            or "would you like to schedule that demo" in lower
            or ("would you like to proceed" in lower and "demo" in lower)
        ):
            recent_assistant_demo_offer = True
            break

    if not recent_assistant_demo_offer:
        return False

    return has_demo_target_context(question, conversation_history)


def _assistant_requested_customer_details(text: str) -> bool:
    lower = (text or "").lower()
    return any(
        phrase in lower
        for phrase in (
            "share your name",
            "share your email",
            "share your phone",
            "share your company",
            "share your contact details",
            "follow up on this request",
            "handled in line with our privacy practices",
            "handled with care",
        )
    )


def should_apply_demo_lead_capture(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    if is_lead_capture_intent_question(question):
        return True
    if _is_affirmative_demo_followup(question, conversation_history):
        return True

    if not conversation_history:
        return False

    recent_user_lead_intent = False
    recent_assistant_asked_for_details = False
    for msg in reversed(conversation_history[-6:]):
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")
        if role == "user" and is_lead_capture_intent_question(content):
            recent_user_lead_intent = True
        if role == "assistant" and _assistant_requested_customer_details(content):
            recent_assistant_asked_for_details = True
        if recent_user_lead_intent and recent_assistant_asked_for_details:
            break

    if not recent_user_lead_intent or not recent_assistant_asked_for_details:
        return False

    if _is_information_request_question(question) and not is_lead_capture_intent_question(question):
        return False

    return bool(
        is_lead_capture_intent_question(question)
        or
        _text_has_contact_details(question)
        or _is_compact_customer_contact_submission(question)
        or _extract_customer_company(question)
        or _text_has_interest_details(question)
        or _extract_customer_name(question)
    )


def get_customer_detail_status(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, bool]:
    status = {
        "name": False,
        "email": False,
        "phone": False,
        "company": False,
        "interest": False,
    }

    texts = [question]
    if conversation_history:
        texts.extend(
            str(msg.get("content") or "")
            for msg in conversation_history
            if msg.get("role") == "user"
        )

    for text in texts:
        status["name"] = status["name"] or bool(_extract_customer_name(text))
        status["email"] = status["email"] or bool(_extract_customer_email(text))
        status["phone"] = status["phone"] or bool(_extract_customer_phone(text))
        status["company"] = status["company"] or bool(_extract_customer_company(text))
        status["interest"] = status["interest"] or _text_has_interest_details(text)

    return status


def get_customer_detail_values(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, str | None]:
    values = {
        "name": None,
        "email": None,
        "phone": None,
        "company": None,
    }

    texts = [question]
    if conversation_history:
        texts.extend(
            str(msg.get("content") or "")
            for msg in conversation_history
            if msg.get("role") == "user"
        )

    for text in texts:
        values["name"] = values["name"] or _extract_customer_name(text)
        values["email"] = values["email"] or _extract_customer_email(text)
        values["phone"] = values["phone"] or _extract_customer_phone(text)
        values["company"] = values["company"] or _extract_customer_company(text)

    return values


def has_customer_contact_details(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> bool:
    status = get_customer_detail_status(question, conversation_history)
    return status["name"] or status["email"] or status["phone"] or status["company"]


def _format_detail_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _answer_already_requests_customer_details(answer: str) -> bool:
    lower = (answer or "").lower()
    return any(
        phrase in lower
        for phrase in (
            "share your name",
            "share your email",
            "share your phone",
            "share your phone number",
            "share your company",
            "share your contact details",
            "could you please share your",
            "could you share your",
            "please provide your",
            "please share your",
        )
    )


def build_required_lead_capture_followup(
    question: str,
    answer: str,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    lead_capture_enabled: bool = False,
) -> str:
    if not lead_capture_enabled:
        return ""
    if not should_apply_demo_lead_capture(question, conversation_history):
        return ""
    if should_clarify_demo_target(question, conversation_history):
        return ""
    if _answer_already_requests_customer_details(answer):
        return ""

    customer_detail_status = get_customer_detail_status(question, conversation_history)
    customer_detail_values = get_customer_detail_values(question, conversation_history)

    missing_details: list[str] = []
    if not customer_detail_status["name"]:
        missing_details.append("name")
    if not customer_detail_status["email"]:
        missing_details.append("email")
    if not customer_detail_status["phone"]:
        missing_details.append("phone number")
    if not customer_detail_status["company"]:
        missing_details.append("company name")
    if not customer_detail_status["interest"]:
        missing_details.append("specific requirements or use case")

    if not missing_details:
        return ""

    provided_fields: list[str] = []
    if customer_detail_status["name"]:
        provided_fields.append("name")
    if customer_detail_status["email"]:
        provided_fields.append("email")
    if customer_detail_status["phone"]:
        provided_fields.append("phone number")
    if customer_detail_status["company"]:
        provided_fields.append("company name")

    missing_text = _format_detail_list(missing_details)
    provided_text = _format_detail_list(provided_fields)
    customer_name = customer_detail_values.get("name")
    name_prefix = f" {customer_name}" if customer_name else ""

    if is_pricing_question(question):
        if provided_text:
            opening = f"Thank you{name_prefix}, we have your {provided_text}. "
        else:
            opening = "To help our team prepare an accurate quote, "
        request = f"Could you please share your {missing_text}?"
    else:
        if provided_text:
            opening = f"Thank you{name_prefix}, we have your {provided_text}. "
        else:
            opening = "To help our team follow up on this request, "
        request = f"Could you please share your {missing_text}?"

    reassurance = (
        "Your details will only be used to follow up on this request and will be handled in line with our privacy practices."
    )
    return f"{opening}{request} {reassurance}".strip()


def build_required_career_followup(
    question: str,
    answer: str,
    context: str,
) -> str:
    if not is_career_question(question):
        return ""

    application_email = _extract_job_application_email(context)
    if not application_email:
        return ""

    answer_lower = (answer or "").lower()
    if application_email.lower() in answer_lower:
        return ""
    if "updated cv" in answer_lower or "send your cv" in answer_lower or "send your updated cv" in answer_lower:
        return ""
    if "resume" in answer_lower and application_email.lower() in answer_lower:
        return ""

    return f"If you're interested in applying, please send your updated CV to {application_email}."


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


def expand_query(
    question: str,
    llm_client=None,
    *,
    allow_llm_rewrite: bool = False,
) -> list[str]:
    """
    Generate up to 4 alternative phrasings:
      1. Original question (always first)
      2. Comparison-specific variants (if detected)
      3. Synonym-expansion variant
      4. LLM-rewritten variants (only when allow_llm_rewrite=True)
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
    if allow_llm_rewrite and llm_client is not None and len(variants) < 3:
        llm_variants = rewrite_query_with_llm(question, llm_client)
        for v in llm_variants:
            if v.lower() not in [x.lower() for x in variants]:
                variants.append(v)

    return variants[:4]


# ── Message builder ────────────────────────────────────────────────────────────

def _format_source_urls(source_urls: list[str] | None) -> str:
    """Source URLs are not shown in responses per RULE 6."""
    return ""


def _extract_grounded_contact_details(context: str) -> dict[str, str | None]:
    if not context:
        return {"email": None, "phone": None}

    cleaned_context = "\n".join(
        line for line in str(context).splitlines()
        if not _CONTACT_PLACEHOLDER_RE.search(line)
    )

    email_match = _CONTACT_EMAIL_RE.search(cleaned_context)
    phone_value: str | None = None
    for match in _CONTACT_PHONE_RE.finditer(cleaned_context):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if len(digits) >= 7:
            phone_value = candidate
            break

    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_value,
    }


def _extract_job_application_email(context: str) -> str | None:
    if not context:
        return None

    cue_re = re.compile(
        r"(?i)\b(job|jobs|career|careers|opening|openings|position|positions|"
        r"vacancy|vacancies|hiring|apply|application|cv|resume|recruitment|hr)\b"
    )

    for raw_line in str(context).splitlines():
        line = raw_line.strip()
        if not line or _CONTACT_PLACEHOLDER_RE.search(line):
            continue
        if not cue_re.search(line):
            continue
        match = _CONTACT_EMAIL_RE.search(line)
        if match:
            return match.group(0)

    for paragraph in re.split(r"\n\s*\n", str(context)):
        text = paragraph.strip()
        if not text or _CONTACT_PLACEHOLDER_RE.search(text):
            continue
        if not cue_re.search(text):
            continue
        match = _CONTACT_EMAIL_RE.search(text)
        if match:
            return match.group(0)

    return None


def build_messages(
    context: str,
    question: str,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    source_urls: list[str] | None = None,
    lead_capture_enabled: bool = False,
    offering_scope: str | None = None,
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

    if lead_capture_enabled:
        merged_system += (
            "\n\n---\n\n"
            "Lead capture guidance:\n"
            "- Ask for customer contact details only in an active high-intent follow-up flow, such as demo or appointment requests, pricing or quote requests, customization, integration, service or support requests, or explicit buying interest.\n"
            "- Ask naturally and only when relevant. Do not ask for contact details in every conversation.\n"
            "- Prefer collecting only the missing details needed for follow-up, such as name, email, phone, company, or requirement summary.\n"
            "- If the user has already shared any contact details, acknowledge that and never ask again for the same field.\n"
            "- If the user shared only part of the details, ask only for the missing ones. Example: if the user already shared name and email, politely ask only for the company name and phone number, and if still needed, the requirement summary.\n"
            "- You may answer the high-intent inquiry and then ask for the customer's missing contact details for follow-up.\n"
            "- If the user moves on to ordinary product, service, or general information questions that are not part of the active high-intent follow-up, answer that new topic directly and do not add any contact-details closing.\n"
            "- When the user has shown interest in a specific product or service, acknowledge that offering by name if it is clear from the question or context.\n"
            "- If you have just offered a demo for a specific product or service and the user replies with a clear confirmation like 'yes', treat that as continuing the same follow-up flow and collect the missing contact details instead of redirecting them to a generic form.\n"
            "- Only when the high-intent follow-up is complete and all key customer details are already shared may you use a short polite confirmation that the team will follow up shortly, followed by an optional sales-team handoff.\n"
            "- For contact handoffs, do NOT introduce the contact block with wording like 'Here are the contact details for scheduling your demo' or 'requesting demo contact this number and email'. Prefer optional quick-connect wording to the sales team instead.\n"
            "- When asking for contact details or confirming follow-up, you may add one short reassurance such as 'Your details will only be used to follow up on this request and will be handled in line with our privacy practices.' Keep it brief and professional.\n"
            "- Keep the request short, professional, and helpful."
        )

    comparison_hint = ""
    if is_comparison_question(question):
        comparison_items = extract_comparison_items(question)
        if len(comparison_items) < 2:
            # Vague comparison — user said "compare products/services" without naming
            # specific items. Ask for clarification instead of hallucinating.
            comparison_hint = (
                "\n\nIMPORTANT: The user has asked to compare, but has NOT specified which "
                "products or services to compare. "
                "Do NOT attempt a comparison. Do NOT guess, invent, or hallucinate any items. "
                "Do NOT list products and compare them on your own. "
                "Instead, respond with ONE short, polite clarification question asking the user "
                "to name the specific products or services they want compared. "
                "Example: 'Which products would you like me to compare? "
                "Please let me know the specific ones you have in mind.' "
                "Keep the response brief and professional — do not add anything else."
            )
        else:
            comparison_hint = (
                "\n\n=== COMPARISON ENGINE — MANDATORY INSTRUCTIONS ===\n"
                "You are acting as a comparison engine. The user has asked to compare two or more items.\n\n"

                "CORE RULES:\n"
                "1. ALWAYS respond to a comparison request. NEVER reject or refuse a comparison.\n"
                "2. EVEN IF the items belong to different categories, still compare them.\n"
                "3. First classify each item:\n"
                "   - Type: Product OR Service\n"
                "   - Category: Identify the category for each item based on the context.\n"
                "4. Comparison logic:\n"
                "   - IF same category → SPEC-BASED comparison: focus on technical attributes and specifications.\n"
                "   - IF different categories → PURPOSE-BASED comparison: focus on use case, function, and when to use each. "
                "DO NOT force technical parameter comparisons across different categories.\n\n"

                "OUTPUT STRUCTURE — generate the response in this exact order:\n\n"

                "### 1. Recommendation (Decision First)\n"
                "Provide a clear, concise recommendation.\n"
                "Format: 'X is better for <use case>, while Y is better for <use case>.'\n\n"

                "### 2. High-Level Summary (ONLY if different categories)\n"
                "Briefly explain the fundamental difference in purpose. Keep it short and clear.\n\n"

                "### 3. Key Differences\n"
                "Use bullet points. Focus on the most important distinctions. Max 4–6 points.\n\n"

                "### 4. Detailed Comparison\n"
                "Present each item as its own labeled block with bullet points. NEVER use a table or markdown table.\n"
                "Format each block exactly like this:\n"
                "[Product/Service Name]\n"
                "- Attribute: value\n"
                "- Attribute: value\n"
                "- Attribute: value\n\n"
                "[Next Product/Service Name]\n"
                "- Attribute: value\n"
                "- Attribute: value\n\n"
                "For same category items: include technical/spec attributes.\n"
                "For different category items: include purpose, usage, behavior, and when to use each.\n\n"
 
                "### 5. Follow-up Question\n"
                "End with a clarifying or decision-driving question.\n"
                "Example: 'Would you like to explore a specific option in more detail, or shall I help you find the best fit for your needs?'\n\n"

                "STYLE GUIDELINES:\n"
                "- Be professional, polite, and positive.\n"
                "- Do NOT use filler words like 'certainly' as an opener.\n"
                "- Be concise but informative.\n"
                "- Avoid unnecessary technical jargon unless required.\n"
                "- Focus on helping the user make a decision.\n\n"

                "IMPORTANT CONSTRAINTS:\n"
                "- Do NOT compare unrelated technical specs across different categories.\n"
                "- Do NOT produce empty or vague comparisons.\n"
                "- Do NOT skip the Recommendation section.\n"
                "- Always prioritize clarity over completeness.\n"
                "- Use ONLY information present in the CONTEXT. Do not invent or hallucinate specifications.\n"
                "- If specific details for one item are missing from context, note that briefly and compare what is available.\n\n"

                "=== END OF COMPARISON ENGINE INSTRUCTIONS ==="
            )

    demo_target_clarification_hint = ""
    if should_clarify_demo_target(question, conversation_history):
        context_scope = offering_scope or detect_context_offering_scope(context)
        if context_scope == "product":
            clarification_example = "Which product would you like to book the appointment or demo for?"
        elif context_scope == "service":
            clarification_example = "Which service would you like to book the appointment or demo for?"
        elif context_scope == "offering":
            clarification_example = "Which product or service would you like to book the appointment or demo for?"
        else:
            clarification_example = "Which product or service would you like to book the appointment or demo for?"
        demo_target_clarification_hint = (
            "\n\nIMPORTANT: The user is asking for a demo or appointment, but they have NOT clearly said which "
            "product, service, or solution they want. "
            "Do NOT assume or infer a specific offering from the context. "
            "Do NOT describe any specific product or service yet. "
            "Instead, ask one short polite clarification question based on the available knowledge scope. "
            f"For this case, use wording like: '{clarification_example}'"
        )

    pricing_hint = ""
    if is_pricing_question(question):
        pricing_subject = detect_pricing_subject(question)
        if pricing_subject:
            subject_phrase = f" for these {pricing_subject}"
        else:
            subject_phrase = ""
        pricing_hint = (
            "\n\nIMPORTANT: This is a pricing-related question. "
            "If the context contains pricing details, answer using only those details. "
            "If pricing details are not present in the context, say that professionally and politely, "
            f"acknowledge that the user is looking for pricing{subject_phrase}, "
            "say that you do not currently have pricing details in the available information, "
            "and explain that pricing can depend on the user's specific requirements. "
            "Then say the sales team can certainly help with accurate pricing aligned to their needs, "
            "and include both the sales phone number and the sales email if those details appear in the context. "
            "If only one of them appears in the context, include whichever is available. "
            "Put those contact details in the next paragraph, not inside the explanatory paragraph. "
            "Show each one on its own labeled line with clear labels such as Email and Phone. "
            "If no real contact details are present in the context, do not output any contact block. "
            "Never output placeholder values such as '[not available]', 'not available', 'N/A', 'NA', 'unknown', or '-'. "
            "If lead capture is active for this conversation, you MUST also ask the customer for the missing follow-up details needed to prepare an accurate quote, such as name, email, phone number, company name, and the specific requirement or use case. "
            "Do not ask again for any field the customer already shared. "
            "For pricing or quote questions with lead capture enabled, do not end the reply without that customer-detail request. "
            "After sharing the contact details, end with one short professional follow-up invitation such as "
            "'If you'd like, I can also help with anything else you need.' or "
            "'Please let me know if there's anything else you'd like to know.' "
            "Use a professional tone such as: "
            "'Thank you for your interest in pricing. At the moment, I don't have access to specific pricing details. "
            "Since pricing can vary depending on your exact requirements, our sales team would be best positioned to provide accurate and tailored information. "
            "I'd be happy to connect you with them to discuss your needs further.'"
        )

    career_hint = ""
    if is_career_question(question):
        job_application_email = _extract_job_application_email(context)
        if job_application_email:
            career_hint = (
                "\n\nIMPORTANT: This is a careers or job-openings question. "
                "After listing or describing the available openings, add one short application note at the end using this exact grounded email address. "
                f"Use wording like: 'If you're interested in applying, please send your updated CV to {job_application_email}.' "
                "If the user shows interest in any open position, include that same application email in the reply. "
                "Do not invent a different email address."
            )

    recommendation_hint = ""
    if is_recommendation_question(question):
        if has_recommendation_requirements(question, conversation_history):
            recommendation_hint = (
                "\n\nIMPORTANT: This appears to be a recommendation or suitability question, and the user's requirements are available. "
                "If you identify a specific product, solution, or service as the best fit, "
                "answer that directly and professionally first. "
                "Then you may add one short polite sentence saying that for a more detailed discussion based on the user's exact requirements, "
                "it would be best to connect with the sales team, who can provide a tailored solution. "
                "If phone and email details are present in the context, include both of them after that handoff. "
                "If only one contact detail is present, include only that one. "
                "Show contact details clearly in the next paragraph on separate labeled lines with clear labels such as Email and Phone. "
                "If no real contact details are present in the context, do not output any contact block. "
                "Never output placeholder values such as '[not available]', 'not available', 'N/A', 'NA', 'unknown', or '-'. "
                "Keep that handoff brief, professional, and natural."
            )
        else:
            recommendation_hint = (
                "\n\nIMPORTANT: This appears to be a recommendation question, but the user's requirements are not clear from the current question or conversation history. "
                "Do NOT recommend a specific product, solution, or service yet. "
                "Instead, respond politely and professionally with a short clarification question asking about the user's needs, use case, goals, or challenges. "
                "Do not include a sales-team handoff in this case. "
                "Do not guess."
            )

    list_scope_hint = ""
    list_category = detect_list_category(question) if is_list_question(question) else None
    catalog_category_hint = ""
    if re.search(r"(?im)^\s*(?:Product|Service) Category\s*:", context):
        catalog_category_hint = (
            "\n\nIMPORTANT: The context contains top-level catalog categories rather than individual products or services. "
            "List those category names clearly and concisely. "
            "Do NOT invent or expand into individual items unless the user asks for a specific category. "
            "End with one short invitation asking which category they would like to explore."
        )
    if list_category == "offering":
        list_scope_hint = (
            "\n\nIMPORTANT: This is a general offerings question. "
            "If the context contains both Product: and Service: items, list BOTH sections together. "
            "Do not narrow the answer to only services or only products because of conversation history. "
            "If only one of those categories exists in context, list only that category."
        )
    elif list_category == "product":
        list_scope_hint = (
            "\n\nIMPORTANT: This question is specifically about products. "
            "List ONLY Product: items from the context. "
            "Do not include Service: items even if they appear elsewhere in the conversation history."
        )
    elif list_category == "service":
        list_scope_hint = (
            "\n\nIMPORTANT: This question is specifically about services. "
            "List ONLY Service: items from the context. "
            "Do not include Product: items even if they appear elsewhere in the conversation history."
        )

    lead_capture_hint = ""
    customer_detail_hint = ""
    grounded_contact_details = _extract_grounded_contact_details(context)
    grounded_contact_hint = (
        "\n\nGROUNDED BUSINESS CONTACT DETAILS FROM CONTEXT:\n"
        f"Email: {grounded_contact_details['email'] or 'missing'}\n"
        f"Phone: {grounded_contact_details['phone'] or 'missing'}\n"
        "If you include a sales or contact block in the final reply, copy only the exact values shown above. "
        "Do not rewrite them, normalize them, or substitute example values. "
        "If Email is marked missing, omit the Email line entirely. "
        "If Phone is marked missing, omit the Phone line entirely. "
        "Never invent a fallback contact number or email address."
    )
    if (
        lead_capture_enabled
        and should_apply_demo_lead_capture(question, conversation_history)
        and not should_clarify_demo_target(question, conversation_history)
    ):
        customer_detail_status = get_customer_detail_status(question, conversation_history)
        customer_detail_values = get_customer_detail_values(question, conversation_history)
        missing_details: list[str] = []
        if not customer_detail_status["name"]:
            missing_details.append("name")
        if not customer_detail_status["email"]:
            missing_details.append("email")
        if not customer_detail_status["phone"]:
            missing_details.append("phone number")
        if not customer_detail_status["company"]:
            missing_details.append("company name")
        if not customer_detail_status["interest"]:
            missing_details.append("the specific requirement or use case")

        if missing_details:
            if len(missing_details) == 1:
                missing_text = missing_details[0]
            elif len(missing_details) == 2:
                missing_text = f"{missing_details[0]} and {missing_details[1]}"
            else:
                missing_text = ", ".join(missing_details[:-1]) + f", and {missing_details[-1]}"

            provided_fields: list[str] = []
            if customer_detail_status["name"]:
                provided_fields.append("name")
            if customer_detail_status["email"]:
                provided_fields.append("email")
            if customer_detail_status["phone"]:
                provided_fields.append("contact number")
            if customer_detail_status["company"]:
                provided_fields.append("company name")

            if len(provided_fields) == 1:
                provided_text = provided_fields[0]
            elif len(provided_fields) == 2:
                provided_text = f"{provided_fields[0]} and {provided_fields[1]}"
            elif not provided_fields:
                provided_text = ""
            else:
                provided_text = ", ".join(provided_fields[:-1]) + f", and {provided_fields[-1]}"

            if is_pricing_question(question):
                if provided_text:
                    acknowledgement = (
                        "thank the customer naturally, acknowledge the details already shared, "
                        f"such as {provided_text}, "
                    )
                else:
                    acknowledgement = "thank the customer naturally, "
                lead_capture_hint = (
                    "\n\nIMPORTANT: This is an active pricing or quote lead-capture follow-up. "
                    "Your final reply MUST collect the missing customer details for quote follow-up. "
                    "If pricing details are not available in the context, say that professionally and explain that pricing depends on the customer's exact requirements. "
                    f"Then {acknowledgement}"
                    f"and politely ask only for the missing details: {missing_text}. "
                    "Do NOT ask again for any detail the customer already shared. "
                    "If the context contains real sales-team contact details, you MUST add a short quick-connect handoff and show those contact details clearly on separate labeled lines after the request for missing customer details. "
                    "If no real sales contact details are present in the context, omit that contact block. "
                    "Add a short reassurance that their details will only be used to follow up on this request and will be handled in line with privacy practices. "
                    "Do not end with only a generic pricing explanation. "
                    "Use a professional style similar to: "
                    "'Thank you for your interest in pricing. To help our team prepare an accurate quote, could you please share your name, email, phone number, company name, and specific requirements or use case? "
                    "Your details will only be used to follow up on this request and will be handled in line with our privacy practices. "
                    "If you would like to discuss your requirements directly with our sales team, please find the contact details below.'"
                )
            else:
                if provided_text:
                    acknowledgement = (
                        "acknowledge the details already shared, "
                        f"such as {provided_text}, "
                    )
                else:
                    acknowledgement = "do not pretend the customer already shared details, "
                lead_capture_hint = (
                    "\n\nIMPORTANT: This is an active high-intent lead-capture follow-up. "
                    "The customer has not yet shared all of the details needed for follow-up. "
                    "Do NOT show the sales team contact details in this case. "
                    "Instead, thank the customer naturally, "
                    f"{acknowledgement}"
                    f"and then politely ask only for the missing details: {missing_text}. "
                    "Do NOT ask again for any detail the customer already shared. "
                    "Add a short reassurance that their details will only be used to follow up on this request and will be handled in line with privacy practices. "
                    "Use a response style similar to: "
                    "'Thank you Sree, we got your name and email. Could you please share your company name and any specific requirements or use cases you have in mind? "
                    "Your details will only be used to follow up on this request and will be handled in line with our privacy practices.'"
                )
        else:
            lead_capture_hint = (
                "\n\nIMPORTANT: This is an active high-intent lead-capture follow-up and the customer has already shared all key follow-up details. "
                "After addressing the request professionally, do NOT ask the customer to share their contact details again. "
                "Instead, thank them for their interest in the relevant product or service, mention that the team will reach out shortly using the contact details shared, "
                "and add a short reassurance that their information will only be used for follow-up and handled in line with privacy practices. "
                "Then, if helpful, add a polite quick-connect handoff such as 'If you would like to discuss your business requirements directly with our sales team, please find the contact details below.' "
                "Do NOT frame the contact block as 'for scheduling your demo' or similar. "
                "Then include the sales email and sales phone if those details are available in the context, each on separate labeled lines such as Email and Phone. "
                "If the context does not contain real sales contact details, do not invent them. "
                "Use a response style similar to: "
                "'Thank you for your interest in ERP-Integrated Field Service Management. We have noted your contact details, and our team will reach out to you shortly. "
                "Your information will only be used for follow-up on this request and will be handled in line with our privacy practices. "
                "If you would like to discuss your business requirements directly with our sales team, please find the contact details below.'"
            )

        detected_lines = [
            f"Name: {customer_detail_values['name'] or 'missing'}",
            f"Email: {customer_detail_values['email'] or 'missing'}",
            f"Phone: {customer_detail_values['phone'] or 'missing'}",
            f"Company: {customer_detail_values['company'] or 'missing'}",
            f"Requirement / use case: {'provided' if customer_detail_status['interest'] else 'missing'}",
        ]
        customer_detail_hint = (
            "\n\nDETECTED CUSTOMER DETAILS FROM THE USER'S MESSAGE/HISTORY:\n"
            + "\n".join(detected_lines)
            + "\nTreat every non-missing value above as already provided by the customer. "
              "Do NOT ask again for any field marked as provided."
        )

    user_prompt = (
        "CONTEXT (answer ONLY from this, nothing else):\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "IMPORTANT: Use ONLY the information in the CONTEXT above. "
        "Do not add general knowledge or explanations. "
        "If the context does not contain enough information to answer fully, "
        "say that warmly and positively without guessing. "
        "If the answer is available, do not begin with 'Thank you for your question'. "
        "Answer directly in a polite and professional way. "
        "Do not mention the CONTEXT, the provided context, or the source material in the final reply. "
        "Answer directly and naturally. "
        "Never say 'contact our team' when you already have an answer. "
        "Only if context has NO relevant info: use a warm fallback and show phone/email if in context."
        f"{comparison_hint}"
        f"{demo_target_clarification_hint}"
        f"{pricing_hint}"
        f"{career_hint}"
        f"{recommendation_hint}"
        f"{list_scope_hint}"
        f"{catalog_category_hint}"
        f"{lead_capture_hint}"
        f"{customer_detail_hint}"
        f"{grounded_contact_hint}"
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