"""
Local embeddings using fastembed (ONNX Runtime, no PyTorch).

- No API key required
- ~150 MB total footprint — safe on Render free tier (512 MB)
- 384-dimensional vectors, already L2-normalised by fastembed
- Model (~24 MB) is downloaded once and cached automatically
"""
import asyncio
from typing import List, Optional

import numpy as np
from fastembed import TextEmbedding

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: Optional[TextEmbedding] = None


def _get_model() -> TextEmbedding:
    """Load model once; reuse on every subsequent call."""
    global _model
    if _model is None:
        _model = TextEmbedding(_MODEL_NAME)
    return _model


def _encode(texts: List[str]) -> np.ndarray:
    model = _get_model()
    # fastembed returns a generator of (384,) float32 arrays, already normalised
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


async def embed_texts(texts: List[str]) -> np.ndarray:
    """Return a (len(texts), 384) float32 array of unit vectors."""
    return await asyncio.to_thread(_encode, texts)


async def embed_query(text: str) -> np.ndarray:
    """Return a single (384,) float32 unit vector."""
    matrix = await embed_texts([text])
    return matrix[0]
