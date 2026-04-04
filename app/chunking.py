from __future__ import annotations

import re

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_MULTISPACE_RE = re.compile(r"\s+")
_GENERIC_CATALOG_TITLE_RE = re.compile(
    r"^(products?|services?|offerings?|solutions?|capabilities|features?|benefits?|"
    r"overview|introduction|summary|about us|contact us|request a demo|demo|faq|faqs|"
    r"sap services)$",
    re.IGNORECASE,
)
_NOISE_LINE_RE = re.compile(
    r"^(home|about|contact|privacy policy|terms(?: of service)?|cookie policy|"
    r"learn more|read more|download now|click here|company overview|core value proposition|"
    r"execution layer capabilities|functional capabilities|sample solutions delivered|"
    r"integrations|sap integration approach|frequently asked questions|industries supported|"
    r"use cases|business benefits|fallback response \(chatbot\))$",
    re.IGNORECASE,
)
_NUMBERED_SECTION_RE = re.compile(r"^\d+\.\s+")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks that always end on a sentence boundary.

    Strategy:
      1. Tokenise into sentences with NLTK (falls back to heuristics if not installed).
      2. Accumulate sentences until the chunk would exceed chunk_size.
      3. When a chunk is full, slide back `overlap` characters of context.
    """
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # A single sentence longer than chunk_size gets char-split (tables, code blocks)
        if sentence_len > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_length = 0
            for sub in _char_chunks(sentence, chunk_size, overlap):
                chunks.append(sub)
            continue

        projected = current_length + (1 if current_sentences else 0) + sentence_len
        if current_sentences and projected > chunk_size:
            chunks.append(" ".join(current_sentences).strip())

            # Overlap: carry trailing sentences whose combined length <= overlap
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) + 1 <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            current_sentences = overlap_sentences
            current_length = overlap_len

        current_sentences.append(sentence)
        current_length += sentence_len + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [c for c in chunks if c]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    if _NLTK_AVAILABLE:
        try:
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    return _heuristic_sentences(text)


def _heuristic_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in parts if s.strip()] or [text.strip()]


def generate_summary_chunk(
    title: str,
    url: str,
    text: str,
    category: str | None = None,
) -> str:
    """
    Build a compact summary chunk for a single product or service page.

    Structure:
        Product: <title>
        or
        Service: <title>
        URL: <url>
        <first meaningful body lines, capped at 250 chars>

    Why line-based extraction instead of sentence splitting:
      Product page titles contain no sentence terminators (.!?) so NLTK and
      heuristic splitters merge the title + URL + first paragraph into one
      giant "sentence" (often 300+ chars).  The old 300-char guard then broke
      immediately on the very first token, leaving summary_body empty for most
      product pages.

      Splitting by lines is more reliable for crawled HTML text: each logical
      block (heading, tagline, feature list) is already on its own line after
      the HTML-to-text extraction step.  We skip lines that duplicate the
      title or look like navigation noise, then stitch the first meaningful
      lines into a compact body.

    No LLM call — purely heuristic, zero extra API cost.
    """
    title_clean = title.strip()
    url_clean   = url.strip()

    # Split the full page text into non-empty lines
    all_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    body_lines: list[str] = []
    total_len  = 0
    skip_set   = {title_clean, url_clean}

    for line in all_lines:
        # Skip lines that duplicate the title or URL (already in parts below)
        if line in skip_set:
            continue
        # Skip very short lines (single words, phone numbers, nav items)
        if len(line) < 20:
            continue
        # Skip lines that look like standalone URLs
        if line.startswith("http://") or line.startswith("https://"):
            continue
        if total_len + len(line) > 250:
            break
        body_lines.append(line)
        total_len += len(line) + 1
        if len(body_lines) >= 3:
            break

    summary_body = " ".join(body_lines).strip()
    label = "Service" if category == "service" else "Product"

    parts: list[str] = []
    if title_clean:
        parts.append(f"{label}: {title_clean}")
    if url_clean:
        parts.append(f"URL: {url_clean}")
    if summary_body:
        parts.append(summary_body)

    return "\n".join(parts)


def generate_catalog_summary_chunks(
    text: str,
    *,
    category: str,
    max_items: int = 20,
) -> list[str]:
    """
    Build item-level Product:/Service: summary chunks from an uploaded document.

    This is used for non-website files where a single document may describe
    multiple products or services. The extractor is heuristic-only and looks for
    heading-like lines or bullet items, then attaches a short description from
    the following lines.
    """
    if category not in {"product", "service"}:
        return []

    label = "Service" if category == "service" else "Product"
    sections = _extract_catalog_sections(text, category=category, max_items=max_items)
    summaries: list[str] = []

    for heading, body in sections:
        parts = [f"{label}: {heading}"]
        if body:
            parts.append(body)
        summary = "\n".join(parts).strip()
        if summary:
            summaries.append(summary)

    return summaries


def _extract_catalog_sections(text: str, *, category: str, max_items: int) -> list[tuple[str, str]]:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return []

    focused_lines = _extract_lines_from_category_section(raw_lines, category=category) or raw_lines
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for raw_line in focused_lines:
        line = _normalize_catalog_line(raw_line)
        if not line:
            continue

        if _is_catalog_heading_candidate(raw_line, line):
            if current_heading:
                sections.append((current_heading, current_body))
            current_heading = line
            current_body = []
            continue

        if current_heading and not _is_catalog_noise_line(line):
            current_body.append(line)

    if current_heading:
        sections.append((current_heading, current_body))

    cleaned_sections: list[tuple[str, str]] = []
    seen_titles: set[str] = set()
    for heading, body_lines in sections:
        normalized_heading = _MULTISPACE_RE.sub(" ", heading).strip().lower()
        if normalized_heading in seen_titles:
            continue
        seen_titles.add(normalized_heading)
        summary_body = _summarize_section_body(body_lines)
        cleaned_sections.append((heading, summary_body))
        if len(cleaned_sections) >= max_items:
            break

    return cleaned_sections


def _extract_lines_from_category_section(raw_lines: list[str], *, category: str) -> list[str]:
    section_markers = {
        "service": (
            "services offered",
            "our services",
            "service offerings",
            "services",
        ),
        "product": (
            "products offered",
            "our products",
            "product offerings",
            "products",
        ),
    }

    markers = section_markers.get(category, ())
    start_idx: int | None = None
    for idx, raw_line in enumerate(raw_lines):
        line = _normalize_catalog_line(raw_line).lower()
        if any(marker == line for marker in markers):
            start_idx = idx + 1
            break

    if start_idx is None:
        return []

    collected: list[str] = []
    for raw_line in raw_lines[start_idx:]:
        normalized = _normalize_catalog_line(raw_line)
        if not normalized:
            continue
        lowered = normalized.lower()
        if _NUMBERED_SECTION_RE.match(raw_line):
            break
        if lowered in {
            "execution layer capabilities",
            "functional capabilities",
            "sample solutions delivered",
            "integrations",
            "sap integration approach",
            "frequently asked questions",
            "industries supported",
            "use cases",
            "business benefits",
            "fallback response (chatbot)",
        }:
            break
        collected.append(raw_line)

    return collected


def _normalize_catalog_line(raw_line: str) -> str:
    line = _BULLET_PREFIX_RE.sub("", raw_line.strip())
    line = line.strip(":- ")
    line = _MULTISPACE_RE.sub(" ", line)
    return line.strip()


def _is_catalog_heading_candidate(raw_line: str, line: str) -> bool:
    if not line:
        return False
    if line.startswith("http://") or line.startswith("https://"):
        return False
    if _GENERIC_CATALOG_TITLE_RE.match(line) or _NOISE_LINE_RE.match(line):
        return False

    word_count = len(line.split())
    if word_count == 0 or word_count > 14:
        return False
    if len(line) < 4 or len(line) > 110:
        return False

    bullet_like = bool(_BULLET_PREFIX_RE.match(raw_line))
    has_sentence_end = line.endswith((".", "?", "!"))
    has_many_commas = line.count(",") >= 2
    titleish = _looks_titleish(line)

    if bullet_like and not has_sentence_end:
        return True
    if titleish and not has_sentence_end and not has_many_commas:
        return True
    return False


def _looks_titleish(line: str) -> bool:
    tokens = [token.strip("():,/|-") for token in line.split() if token.strip("():,/|-")]
    if not tokens:
        return False

    titleish_tokens = 0
    for token in tokens:
        if token.isupper() and len(token) > 1:
            titleish_tokens += 1
        elif token[:1].isupper():
            titleish_tokens += 1
        elif any(char.isdigit() for char in token):
            titleish_tokens += 1

    ratio = titleish_tokens / len(tokens)
    return ratio >= 0.5


def _is_catalog_noise_line(line: str) -> bool:
    if not line:
        return True
    if line.startswith("http://") or line.startswith("https://"):
        return True
    return bool(_NOISE_LINE_RE.match(line))


def _summarize_section_body(body_lines: list[str], limit: int = 240) -> str:
    picked: list[str] = []
    total_len = 0
    for line in body_lines:
        if len(line) < 20:
            continue
        if total_len + len(line) > limit:
            break
        picked.append(line)
        total_len += len(line) + 1
        if len(picked) >= 3:
            break
    return " ".join(picked).strip()


def _char_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
