from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        ideal_end = min(start + chunk_size, text_length)
        end = _find_chunk_end(text, start, ideal_end)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        if next_start <= start:
            next_start = min(start + chunk_size, text_length)
        start = next_start

    return chunks


def _find_chunk_end(text: str, start: int, ideal_end: int) -> int:
    if ideal_end >= len(text):
        return len(text)

    search_window_start = min(start + max((ideal_end - start) // 2, 1), ideal_end)
    for separator in ("\n\n", ". ", "? ", "! ", "\n", " "):
        split_at = text.rfind(separator, search_window_start, ideal_end)
        if split_at != -1:
            return split_at + len(separator)

    return ideal_end
