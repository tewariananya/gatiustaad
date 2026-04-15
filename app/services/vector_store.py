"""
BM25 keyword retrieval scoped to a single SessionData object.
Replaces FAISS + fastembed — uses ~5 MB vs ~450 MB, safe on Render free tier.
"""
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.models import Confidence
from app.services.chunker import TextChunk
from app.session_store import ChunkRecord, SessionData

# BM25 scores are unbounded; anything above this threshold is a confident match
DIRECT_MATCH_THRESHOLD = 1.0


def score_to_confidence(score: float) -> Confidence:
    return Confidence.direct_match if score >= DIRECT_MATCH_THRESHOLD else Confidence.best_match


async def add_chunks(
    session: SessionData,
    chunks: List[TextChunk],
    document_name: str,
) -> int:
    """Add chunks to session and rebuild the BM25 index."""
    if not chunks:
        return 0

    start_id = len(session.chunks)
    for i, chunk in enumerate(chunks):
        session.chunks.append(ChunkRecord(
            chunk_id=start_id + i,
            document_name=document_name,
            page_number=chunk.page_number,
            section=chunk.section,
            text=chunk.text,
            token_count=chunk.token_count,
        ))

    # Rebuild BM25 over ALL chunks so multi-doc sessions work correctly
    tokenized = [c.text.lower().split() for c in session.chunks]
    session.bm25 = BM25Okapi(tokenized)

    return len(chunks)


def search(
    session: SessionData,
    query: str,
    top_k: int = 5,
) -> Tuple[List[ChunkRecord], float]:
    """Return (top_k chunks, top_score). Pure sync — no embeddings needed."""
    if not session.chunks or session.bm25 is None:
        return [], 0.0

    tokens = query.lower().split()
    scores = session.bm25.get_scores(tokens)

    k = min(top_k, len(session.chunks))
    top_indices = scores.argsort()[-k:][::-1]

    # Only return chunks with a positive score
    results = [session.chunks[i] for i in top_indices if scores[i] > 0]
    top_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0

    return results, top_score
