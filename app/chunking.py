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