"""
category_detector.py
Detects the content category of crawled pages or uploaded documents based on
URL patterns, titles, and text content.

Categories: product | service | pricing | general
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

_PRICING_URL_PATTERNS = re.compile(
    r"/(pricing|plans?|subscription|buy|purchase|checkout|billing|cost|rates?)/",
    re.IGNORECASE,
)
_PRODUCT_URL_PATTERNS = re.compile(
    r"/(product|products|item|items|catalog|catalogue|shop|store)/",
    re.IGNORECASE,
)
_SERVICE_URL_PATTERNS = re.compile(
    r"/(service|services|offering|offerings|feature|features|support|consulting|implementation)/",
    re.IGNORECASE,
)
_GENERAL_URL_PATTERNS = re.compile(
    r"/(about|contact|legal|terms|privacy|cookie|cookies|data-protection|"
    r"gdpr|career|careers|blog|blogs|news|press|case-stud(?:y|ies)|"
    r"documentation|docs|faq|faqs|industry|industries|request-demo|demo|"
    r"download|downloads|company|history|leadership|team|platform|"
    r"management-team|management-board|our-management|senior-management|"
    r"client|clients|location|locations)/",
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
_GENERAL_TITLE_KEYWORDS = re.compile(
    r"\b(about|contact|get in touch|terms|privacy|policy|data protection|"
    r"cookie policy|careers|blog|news|press|documentation|faq|request a demo|demo|"
    r"product finder|service finder|downloads?|company history|history|leadership|"
    r"management team|senior management|top management|our management|"
    r"executive management|management board|"
    r"our clients|clients|locations?)\b",
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

    parsed_url = urlparse(url_lower)
    url_path = parsed_url.path or "/"

    if url_path in {"/product", "/products", "/service", "/services", "/industry", "/industries"}:
        return "general"

    # Strong path-based signal checked FIRST — before any title-keyword override.
    # A page under /products/ or /services/ is always a product/service page
    # regardless of what words appear in its title (e.g. "Field Service Management").
    if "/products/" in url_path:
        return "product"
    if "/services/" in url_path:
        return "service"

    # Non-catalog pages often mention products/services in legal or company copy,
    # but they should never appear as product/service listings.
    if _GENERAL_URL_PATTERNS.search(url_path) or _GENERAL_TITLE_KEYWORDS.search(title_lower):
        return "general"

    # --- URL-based detection (high confidence) ---
    if _PRICING_URL_PATTERNS.search(url_path):
        scores["pricing"] += 10
    if _PRODUCT_URL_PATTERNS.search(url_path):
        scores["product"] += 10
    if _SERVICE_URL_PATTERNS.search(url_path):
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

    # For uploaded files (no URL), we have no URL path signal at all so the score
    # is driven purely by text/title keywords. Lower the threshold from 3 → 1 so
    # a document that clearly mentions "service" or "product" even once is not
    # silently downgraded to "general" and stripped of its summary chunks.
    min_score = 1 if not url else 3
    if scores[best_category] >= min_score:
        return best_category

    return "general"


def detect_document_category(
    title: str = "",
    text: str = "",
) -> str:
    """
    Return the best-fit category for an uploaded file or manual document.

    Uploaded files do not have a useful URL path, so classification relies on
    the filename/title and the extracted content.
    """
    return detect_page_category(url="", title=title, text=text)