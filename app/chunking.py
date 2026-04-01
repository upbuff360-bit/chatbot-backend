from __future__ import annotations

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


def generate_summary_chunk(title: str, url: str, text: str) -> str:
    """
    Build a compact summary chunk for a single product or service page.

    Structure:
        Product: <title>
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

    parts: list[str] = []
    if title_clean:
        parts.append(f"Product: {title_clean}")
    if url_clean:
        parts.append(f"URL: {url_clean}")
    if summary_body:
        parts.append(summary_body)

    return "\n".join(parts)


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