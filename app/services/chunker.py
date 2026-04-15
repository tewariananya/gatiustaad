"""
Character-based text chunker — no external dependencies.

Approximation: 1 token ≈ 4 characters for English/Hindi text.
At 600 tokens: 2400 chars per chunk, 400 chars overlap.
Splits at word boundaries so chunks never cut mid-word.
"""
from dataclasses import dataclass
from typing import List

from app.services.document_processor import PageContent

_CHARS_PER_TOKEN = 4


@dataclass
class TextChunk:
    text: str
    page_number: int
    section: str
    token_count: int


def chunk_pages(
    pages: List[PageContent],
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[TextChunk]:
    """Return character-bounded chunks with page/section attribution."""
    chunk_chars = chunk_size * _CHARS_PER_TOKEN   # 2400
    overlap_chars = overlap * _CHARS_PER_TOKEN     # 400
    step = chunk_chars - overlap_chars             # 2000

    # Build flat character stream with per-character source metadata
    all_text: str = ""
    char_meta: List[tuple] = []  # (page_num, section) per character

    for page in pages:
        all_text += page.text
        char_meta.extend([(page.page_num, page.section)] * len(page.text))

    if not all_text:
        return []

    chunks: List[TextChunk] = []
    start = 0

    while start < len(all_text):
        end = min(start + chunk_chars, len(all_text))

        # Extend to next word boundary (max 100 extra chars) so we don't cut mid-word
        if end < len(all_text):
            space = all_text.find(" ", end)
            if space != -1 and space - end < 100:
                end = space

        chunk_text = all_text[start:end].strip()
        if chunk_text:
            page_num, section = char_meta[start]
            chunks.append(TextChunk(
                text=chunk_text,
                page_number=page_num,
                section=section,
                token_count=len(chunk_text) // _CHARS_PER_TOKEN,
            ))

        if end >= len(all_text):
            break
        start += step

    return chunks
