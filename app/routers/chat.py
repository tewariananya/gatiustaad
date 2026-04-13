"""
POST /chat
    Send a message, receive a streaming Server-Sent Events response.

POST /chat/{message_id}/feedback
    Thumbs-up / thumbs-down on any bot message.

── SSE event sequence ────────────────────────────────────────────────────────

  data: {"type": "start",    "thread_id": "…", "message_id": "…",
                             "confidence": "direct_match"|"best_match",
                             "citations": [{page_number, section, snippet}, …]}

  data: {"type": "token",    "content": "…"}   ← one per text delta

  data: {"type": "done"}

  data: {"type": "error",    "message": "…"}   ← only on failure

── Two-step image flow (PRD §3.2) ────────────────────────────────────────────

  If image_base64 is provided:
    1. Vision pass  — Claude describes what it observes (describe_image)
    2. Description is merged into the retrieval query
    3. Final generation receives both the manual context and the original image
"""
import json
import uuid
from datetime import datetime
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.models import ChatRequest, FeedbackRequest, MessageOut
from app.services.embeddings import embed_query
from app.services.llm import (
    RAG_SYSTEM,
    build_citations,
    build_rag_user_content,
    describe_image,
)
from app.services.vector_store import score_to_confidence, search
from app.session_store import ChatMessage, session_store

router = APIRouter()

_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Headers required for SSE to work through proxies / nginx
_SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",   # disable nginx proxy buffering
    "Connection": "keep-alive",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _auto_thread_name(message: str) -> str:
    """Use the first sentence (≤ 60 chars) or truncate at 48 chars."""
    msg = message.strip()
    for end_char in ["?", ".", "!"]:
        idx = msg.find(end_char)
        if 0 < idx <= 60:
            return msg[: idx + 1]
    if len(msg) <= 48:
        return msg
    return msg[:48].rsplit(" ", 1)[0] + "…"


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("/chat", summary="Ask a troubleshooting question (streaming SSE)")
async def chat(request: ChatRequest) -> StreamingResponse:
    # ── Validate ──────────────────────────────────────────────────────────────
    if request.image_base64 and request.image_media_type not in _ALLOWED_IMAGE_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"image_media_type must be one of {sorted(_ALLOWED_IMAGE_MIME)}.",
        )

    session = await session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    if session.index is None or session.index.ntotal == 0:
        raise HTTPException(status_code=422, detail="No documents uploaded to this session yet.")

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # ── Two-step image flow ───────────────────────────────────────────────────
    retrieval_query = request.message
    if request.image_base64 and request.image_media_type:
        image_description = await describe_image(
            request.image_base64, request.image_media_type, anthropic_client
        )
        retrieval_query = f"{request.message}\n\nImage observation: {image_description}"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    query_vec = await embed_query(retrieval_query, openai_client)
    chunks, top_score = search(session, query_vec, top_k=settings.top_k)
    if not chunks:
        raise HTTPException(status_code=422, detail="No relevant context found for the query.")

    confidence = score_to_confidence(top_score)
    citations = build_citations(chunks)
    user_content = build_rag_user_content(
        retrieval_query, chunks, request.image_base64, request.image_media_type
    )

    # ── Thread + message IDs ──────────────────────────────────────────────────
    thread = session.get_or_create_thread(request.thread_id)
    user_msg_id = str(uuid.uuid4())
    bot_msg_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Persist user message before streaming begins
    thread.messages.append(ChatMessage(
        message_id=user_msg_id,
        role="user",
        content=request.message,
        timestamp=now,
    ))
    # Auto-name thread from first user message
    if thread.name == "New chat":
        thread.name = _auto_thread_name(request.message)

    # ── SSE generator ─────────────────────────────────────────────────────────
    async def event_stream() -> AsyncIterator[str]:
        accumulated: list[str] = []
        try:
            # 1. Start event — metadata the client needs before any tokens arrive
            yield _sse({
                "type": "start",
                "thread_id": thread.thread_id,
                "message_id": bot_msg_id,
                "confidence": confidence.value,
                "citations": [c.model_dump() for c in citations],
            })

            # 2. Token stream
            async with anthropic_client.messages.stream(
                model=settings.claude_model,
                max_tokens=1024,
                system=RAG_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                async for text in stream.text_stream:
                    accumulated.append(text)
                    yield _sse({"type": "token", "content": text})

            # 3. Done
            yield _sse({"type": "done"})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

        finally:
            # Persist bot message regardless of whether client disconnected
            full_text = "".join(accumulated)
            if full_text:
                thread.messages.append(ChatMessage(
                    message_id=bot_msg_id,
                    role="assistant",
                    content=full_text,
                    timestamp=datetime.utcnow(),
                    citations=citations,
                ))
                session.query_count += 1

    return StreamingResponse(event_stream(), headers=_SSE_HEADERS)


# ── POST /chat/{message_id}/feedback ─────────────────────────────────────────

@router.post(
    "/chat/{message_id}/feedback",
    summary="Submit thumbs-up / thumbs-down feedback on a bot message",
)
async def submit_feedback(message_id: str, body: FeedbackRequest) -> dict:
    """Scans all sessions/threads for the message_id and records the signal."""
    for session in session_store.all_sessions():
        for thread in session.threads.values():
            for msg in thread.messages:
                if msg.message_id == message_id and msg.role == "assistant":
                    msg.feedback = body.signal.value
                    return {
                        "status": "ok",
                        "message_id": message_id,
                        "signal": body.signal.value,
                    }

    raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found.")
