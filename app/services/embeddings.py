"""
Local embeddings using sentence-transformers (all-MiniLM-L6-v2).

- No API key required
- Runs entirely on CPU
- 384-dimensional vectors, L2-normalised for cosine similarity via FAISS IndexFlatIP
- Model is downloaded once on first use (~90 MB) and cached automatically
"""
import asyncio
from typing import List

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _encode(texts: List[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


async def embed_texts(texts: List[str]) -> np.ndarray:
    """Return a (len(texts), 384) float32 array of unit vectors."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _encode, texts)


async def embed_query(text: str) -> np.ndarray:
    """Return a single (384,) float32 unit vector."""
    matrix = await embed_texts([text])
    return matrix[0]
