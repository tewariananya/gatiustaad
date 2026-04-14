"""
POST /chat          — streaming SSE response
POST /chat/{id}/feedback — thumbs up/down
"""
import json
import uuid
from datetime import datetime
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import ChatRequest, FeedbackRequest
from app.services.embeddings import embed_query
from app.services.llm import RAG_SYSTEM, build_citations, build_rag_user_content, describe_image
from app.services.vector_store import score_to_confidence, search
from app.session_store import ChatMessage, session_store

router = APIRouter()

_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _auto_thread_name(message: str) -> str:
    msg = message.strip()
    for end_char in ["?", ".", "!"]:
        idx = msg.find(end_char)
        if 0 < idx <= 60:
            return msg[: idx + 1]
    if len(msg) <= 48:
        return msg
    return msg[:48].rsplit(" ", 1)[0] + "…"


@router.post("/chat", summary="Ask a troubleshooting question (streaming SSE)")
async def chat(request: ChatRequest) -> StreamingResponse:
    if request.image_base64 and request.image_media_type not in _ALLOWED_IMAGE_MIME:
        raise HTTPException(400, detail=f"image_media_type must be one of {sorted(_ALLOWED_IMAGE_MIME)}.")

    session = await session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(404, detail=f"Session '{request.session_id}' not found.")
    if session.index is None or session.index.ntotal == 0:
        raise HTTPException(422, detail="No documents uploaded to this session yet.")

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # ── Two-step image flow ───────────────────────────────────────────────────
    retrieval_query = request.message
    if request.image_base64 and request.image_media_type:
        image_description = await describe_image(
            request.image_base64, request.image_media_type, anthropic_client
        )
        retrieval_query = f"{request.message}\n\nImage observation: {image_description}"

    # ── Retrieval (local embeddings) ──────────────────────────────────────────
    query_vec = await embed_query(retrieval_query)
    chunks, top_score = search(session, query_vec, top_k=settings.top_k)
    if not chunks:
        raise HTTPException(422, detail="No relevant context found.")

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

    thread.messages.append(ChatMessage(
        message_id=user_msg_id, role="user",
        content=request.message, timestamp=now,
    ))
    if thread.name == "New chat":
        thread.name = _auto_thread_name(request.message)

    # ── SSE stream ────────────────────────────────────────────────────────────
    async def event_stream() -> AsyncIterator[str]:
        accumulated: list[str] = []
        try:
            yield _sse({
                "type": "start",
                "thread_id": thread.thread_id,
                "message_id": bot_msg_id,
                "confidence": confidence.value,
                "citations": [c.model_dump() for c in citations],
            })

            async with anthropic_client.messages.stream(
                model=settings.claude_model,
                max_tokens=1024,
                system=RAG_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                async for text in stream.text_stream:
                    accumulated.append(text)
                    yield _sse({"type": "token", "content": text})

            yield _sse({"type": "done"})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            full_text = "".join(accumulated)
            if full_text:
                thread.messages.append(ChatMessage(
                    message_id=bot_msg_id, role="assistant",
                    content=full_text, timestamp=datetime.utcnow(),
                    citations=citations,
                ))
                session.query_count += 1

    return StreamingResponse(event_stream(), headers=_SSE_HEADERS)


@router.post("/chat/{message_id}/feedback", summary="Thumbs-up / down on a bot message")
async def submit_feedback(message_id: str, body: FeedbackRequest) -> dict:
    for session in session_store.all_sessions():
        for thread in session.threads.values():
            for msg in thread.messages:
                if msg.message_id == message_id and msg.role == "assistant":
                    msg.feedback = body.signal.value
                    return {"status": "ok", "message_id": message_id, "signal": body.signal.value}
    raise HTTPException(404, detail=f"Message '{message_id}' not found.")
