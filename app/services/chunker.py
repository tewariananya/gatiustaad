"""
Token-aware chunker with overlap.

Strategy: flatten all pages into a single token stream while recording
(page_num, section) per token, then slide a window of `chunk_size` tokens
stepping by (chunk_size - overlap) each time.  Each chunk is labelled with
the metadata of its *first* token, which gives stable, deterministic citations.
"""
from dataclasses import dataclass
from typing import List

import tiktoken

from app.services.document_processor import PageContent

# cl100k_base is the tokeniser used by text-embedding-3-small
_ENC = tiktoken.get_encoding("cl100k_base")


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
    """Return token-bounded chunks with page/section attribution."""
    # Build flat token list + parallel metadata array
    all_tokens: List[int] = []
    token_meta: List[tuple[int, str]] = []  # (page_num, section) per token

    for page in pages:
        tokens = _ENC.encode(page.text)
        all_tokens.extend(tokens)
        token_meta.extend([(page.page_num, page.section)] * len(tokens))

    if not all_tokens:
        return []

    step = chunk_size - overlap
    chunks: List[TextChunk] = []
    start = 0

    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        chunk_tokens = all_tokens[start:end]
        page_num, section = token_meta[start]

        chunks.append(TextChunk(
            text=_ENC.decode(chunk_tokens),
            page_number=page_num,
            section=section,
            token_count=len(chunk_tokens),
        ))

        if end >= len(all_tokens):
            break
        start += step

    return chunks
