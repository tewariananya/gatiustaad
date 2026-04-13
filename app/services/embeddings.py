"""
Thin wrapper around OpenAI text-embedding-3-small.

Vectors are L2-normalised so they can be used directly with FAISS IndexFlatIP
(inner product == cosine similarity for unit vectors).
"""
from typing import List

import numpy as np
from openai import AsyncOpenAI

from app.config import settings

_BATCH_SIZE = 100  # OpenAI embeds up to 2048 inputs; 100 keeps requests small


async def embed_texts(texts: List[str], client: AsyncOpenAI) -> np.ndarray:
    """Return a (len(texts), embedding_dim) float32 array of unit vectors."""
    all_vecs: List[np.ndarray] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
        all_vecs.append(vecs)

    matrix = np.vstack(all_vecs)
    # Normalise rows → cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


async def embed_query(text: str, client: AsyncOpenAI) -> np.ndarray:
    """Return a single (embedding_dim,) float32 unit vector."""
    matrix = await embed_texts([text], client)
    return matrix[0]
