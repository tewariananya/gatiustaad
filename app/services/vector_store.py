"""
FAISS operations scoped to a single SessionData object.

IndexFlatIP with L2-normalised vectors → inner product == cosine similarity.
"""
from typing import List, Tuple

import numpy as np
from openai import AsyncOpenAI

from app.models import Confidence
from app.services.chunker import TextChunk
from app.services.embeddings import embed_texts
from app.session_store import ChunkRecord, SessionData

# Cosine similarity threshold above which we call it a "direct match"
DIRECT_MATCH_THRESHOLD = 0.70


def score_to_confidence(score: float) -> Confidence:
    return Confidence.direct_match if score >= DIRECT_MATCH_THRESHOLD else Confidence.best_match


async def add_chunks(
    session: SessionData,
    chunks: List[TextChunk],
    document_name: str,
    openai_client: AsyncOpenAI,
) -> int:
    """Embed chunks, add to session index, return count added."""
    if not chunks:
        return 0

    session.init_index()
    vectors = await embed_texts([c.text for c in chunks], openai_client)

    start_id = len(session.chunks)
    session.index.add(vectors)  # type: ignore[union-attr]

    for i, chunk in enumerate(chunks):
        session.chunks.append(ChunkRecord(
            chunk_id=start_id + i,
            document_name=document_name,
            page_number=chunk.page_number,
            section=chunk.section,
            text=chunk.text,
            token_count=chunk.token_count,
        ))

    return len(chunks)


def search(
    session: SessionData,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> Tuple[List[ChunkRecord], float]:
    """
    Return (chunks, top_score).
    top_score is the cosine similarity of the best hit (0.0 if index is empty).
    """
    if session.index is None or session.index.ntotal == 0:
        return [], 0.0

    k = min(top_k, session.index.ntotal)
    query_2d = query_vector.reshape(1, -1)
    scores, indices = session.index.search(query_2d, k)  # type: ignore[union-attr]

    chunks = [
        session.chunks[idx]
        for idx in indices[0]
        if 0 <= idx < len(session.chunks)
    ]
    top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
    return chunks, top_score
