"""
Claude integration helpers used by the streaming chat router.

Public API
----------
RAG_SYSTEM              system prompt constant
describe_image()        vision pass → text description
build_rag_user_content() builds the Claude `messages[0].content` list
build_citations()       converts ChunkRecords → Citation models
"""
from typing import List, Optional

from anthropic import AsyncAnthropic

from app.config import settings
from app.models import Citation
from app.session_store import ChunkRecord

# ── System prompts ────────────────────────────────────────────────────────────

RAG_SYSTEM = """\
You are GatiUstaad, an expert bike mechanic assistant. \
Your knowledge comes EXCLUSIVELY from the manual excerpts provided as context.

STRICT RULES:
1. Answer ONLY using information found in the provided context. Do NOT use any external knowledge.
2. If the answer is not present in the context, respond with exactly:
   "I couldn't find information about this in your manual. This topic may not be covered, \
or you could try rephrasing your question."
3. After every factual claim, append a source tag: [Page X] or [Page X, Section: Y].
4. Be precise and technical — users are troubleshooting and repairing their bikes.
5. Never speculate, infer, or extrapolate beyond what the manual explicitly states.\
"""

_VISION_SYSTEM = """\
You are a vehicle diagnostic assistant. \
Describe ONLY what you observe in the image — be specific and technical. \
Focus on visible bike components, symptoms (smoke, leaks, warning lights, damage, etc.). \
Output a concise 1-3 sentence description. No greetings, no advice.\
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: List[ChunkRecord]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        sec = f", Section: {c.section}" if c.section else ""
        header = f"[Context {i} | Page {c.page_number}{sec} | Source: {c.document_name}]"
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_citations(chunks: List[ChunkRecord]) -> List[Citation]:
    return [
        Citation(
            page_number=c.page_number,
            section=c.section or "General",
            snippet=c.text[:200] + ("…" if len(c.text) > 200 else ""),
        )
        for c in chunks
    ]


def build_rag_user_content(
    query: str,
    chunks: List[ChunkRecord],
    image_base64: Optional[str] = None,
    image_media_type: Optional[str] = None,
) -> list:
    """
    Build the `content` list for the Claude user turn.
    Optionally prepends an image block for visual queries.
    """
    context_text = _build_context(chunks)
    content: list = []

    if image_base64 and image_media_type:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_base64,
            },
        })

    content.append({
        "type": "text",
        "text": f"MANUAL CONTEXT:\n\n{context_text}\n\n---\n\nQUESTION: {query}",
    })
    return content


# ── Vision pass ───────────────────────────────────────────────────────────────

async def describe_image(
    image_base64: str,
    image_media_type: str,
    client: AsyncAnthropic,
) -> str:
    """Step 1 of the two-step image flow: describe what is visible in the image."""
    response = await client.messages.create(
        model=settings.claude_model,
        max_tokens=256,
        system=_VISION_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": "Describe what you see in this image."},
            ],
        }],
    )
    return response.content[0].text.strip()
