"""
category_detector.py
Detects the content category of a crawled web page based on URL patterns,
page title, and text content.

Categories: product | service | pricing | general
"""

from __future__ import annotations

import re

_PRICING_URL_PATTERNS = re.compile(
    r"/(pricing|plans?|subscription|buy|purchase|checkout|billing|cost|rates?)/",
    re.IGNORECASE,
)
_PRODUCT_URL_PATTERNS = re.compile(
    r"/(product|products|item|items|catalog|catalogue|shop|store|solution|solutions)/",
    re.IGNORECASE,
)
_SERVICE_URL_PATTERNS = re.compile(
    r"/(service|services|offering|offerings|feature|features|support|consulting|implementation)/",
    re.IGNORECASE,
)

_PRICING_KEYWORDS = re.compile(
    r"\b(pricing|price|prices|plan|plans|subscription|per month|per year|per user|"
    r"free trial|billed annually|billed monthly|upgrade|downgrade|tier|tiers|"
    r"\$\d|\€\d|₹\d|usd|inr|discount|coupon|quote)\b",
    re.IGNORECASE,
)
_PRODUCT_KEYWORDS = re.compile(
    r"\b(product|products|specification|specifications|model|models|variant|variants|"
    r"sku|add to cart|buy now|in stock|out of stock|inventory|catalog|catalogue|"
    r"datasheet|brochure|part number)\b",
    re.IGNORECASE,
)
_SERVICE_KEYWORDS = re.compile(
    r"\b(service|services|we offer|our team|consulting|implementation|integration|"
    r"maintenance|support|managed|professional|expertise|onboarding|training|"
    r"deployment|migration|sla|service level)\b",
    re.IGNORECASE,
)


def detect_page_category(
    url: str = "",
    title: str = "",
    text: str = "",
) -> str:
    """
    Return the best-fit category for a crawled page.

    Priority order: pricing > product > service > general
    Uses URL path, page title, and body text for scoring.
    """
    url_lower = (url or "").lower()
    title_lower = (title or "").lower()
    # Only use first 2000 chars of text for performance
    text_sample = (text or "")[:2000]

    scores: dict[str, int] = {"pricing": 0, "product": 0, "service": 0}

    # --- URL-based detection (high confidence) ---
    if _PRICING_URL_PATTERNS.search(url_lower):
        scores["pricing"] += 10
    if _PRODUCT_URL_PATTERNS.search(url_lower):
        scores["product"] += 10
    if _SERVICE_URL_PATTERNS.search(url_lower):
        scores["service"] += 10

    # --- Title-based detection (medium confidence) ---
    pricing_in_title = len(_PRICING_KEYWORDS.findall(title_lower))
    product_in_title = len(_PRODUCT_KEYWORDS.findall(title_lower))
    service_in_title = len(_SERVICE_KEYWORDS.findall(title_lower))

    scores["pricing"] += pricing_in_title * 5
    scores["product"] += product_in_title * 5
    scores["service"] += service_in_title * 5

    # --- Text-based detection (lower confidence) ---
    pricing_in_text = len(_PRICING_KEYWORDS.findall(text_sample))
    product_in_text = len(_PRODUCT_KEYWORDS.findall(text_sample))
    service_in_text = len(_SERVICE_KEYWORDS.findall(text_sample))

    scores["pricing"] += pricing_in_text
    scores["product"] += product_in_text
    scores["service"] += service_in_text

    best_category = max(scores, key=lambda k: scores[k])

    # Only assign a specific category if there's a meaningful signal
    if scores[best_category] >= 3:
        return best_category

    return "general"