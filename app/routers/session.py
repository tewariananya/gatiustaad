"""
Session and thread management endpoints (PRD §5.3):

  GET    /sessions                               list all active sessions
  GET    /sessions/{id}/info                     full session detail
  PATCH  /sessions/{id}                          update bike name / make / model / year
  DELETE /sessions/{id}                          remove session and its vector store
  GET    /sessions/{id}/manual                   return full extracted manual text
  GET    /sessions/{id}/threads                  list chat threads
  PATCH  /sessions/{id}/threads/{tid}            rename a thread
  DELETE /sessions/{id}/threads/{tid}            delete a thread
  GET    /sessions/{id}/threads/{tid}/messages   full message history for a thread
"""
from fastapi import APIRouter, HTTPException

from app.models import (
    BikeUpdateRequest,
    MessageOut,
    SessionInfo,
    SessionSummary,
    ThreadOut,
    ThreadRenameRequest,
)
from app.session_store import session_store

router = APIRouter(prefix="/sessions")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thread_out(thread) -> ThreadOut:
    return ThreadOut(
        thread_id=thread.thread_id,
        name=thread.name,
        created_at=thread.created_at,
        message_count=len(thread.messages),
    )


async def _get_or_404(session_id: str):
    session = await session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


# ── Session list ──────────────────────────────────────────────────────────────

@router.get("", response_model=list[SessionSummary], summary="List all active sessions")
async def list_sessions() -> list[SessionSummary]:
    return [
        SessionSummary(
            session_id=s.session_id,
            created_at=s.created_at,
            bike_info=s.bike_info,
            document_count=len(s.documents),
            thread_count=len(s.threads),
            total_chunks=len(s.chunks),
        )
        for s in session_store.all_sessions()
    ]


# ── Session detail ────────────────────────────────────────────────────────────

@router.get("/{session_id}/info", response_model=SessionInfo, summary="Get full session detail")
async def get_session_info(session_id: str) -> SessionInfo:
    s = await _get_or_404(session_id)
    return SessionInfo(
        session_id=s.session_id,
        created_at=s.created_at,
        bike_info=s.bike_info,
        documents=s.documents,
        sections=s.sections,
        total_chunks=len(s.chunks),
        total_queries=s.query_count,
        threads=[_thread_out(t) for t in s.threads.values()],
    )


# ── Update bike metadata ──────────────────────────────────────────────────────

@router.patch("/{session_id}", summary="Update bike name / make / model / year")
async def update_session(session_id: str, body: BikeUpdateRequest) -> dict:
    s = await _get_or_404(session_id)
    if body.name is not None:
        s.bike_info.name = body.name
    if body.make is not None:
        s.bike_info.make = body.make
    if body.model is not None:
        s.bike_info.model = body.model
    if body.year is not None:
        s.bike_info.year = body.year
    return {"status": "ok", "session_id": session_id, "bike_info": s.bike_info}


# ── Delete session ────────────────────────────────────────────────────────────

@router.delete("/{session_id}", summary="Delete a session and its vector store")
async def delete_session(session_id: str) -> dict:
    deleted = await session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}


# ── Manual preview ────────────────────────────────────────────────────────────

@router.get("/{session_id}/manual", summary="Return the full extracted manual text")
async def get_manual(session_id: str) -> dict:
    s = await _get_or_404(session_id)
    if not s.manual_text.strip():
        raise HTTPException(status_code=404, detail="No manual text available for this session.")
    return {
        "session_id": session_id,
        "manual_text": s.manual_text,
        "sections": s.sections,
    }


# ── Thread list ───────────────────────────────────────────────────────────────

@router.get("/{session_id}/threads", response_model=list[ThreadOut], summary="List chat threads")
async def list_threads(session_id: str) -> list[ThreadOut]:
    s = await _get_or_404(session_id)
    return [_thread_out(t) for t in s.threads.values()]


# ── Rename thread ─────────────────────────────────────────────────────────────

@router.patch("/{session_id}/threads/{thread_id}", summary="Rename a chat thread")
async def rename_thread(session_id: str, thread_id: str, body: ThreadRenameRequest) -> dict:
    s = await _get_or_404(session_id)
    thread = s.threads.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    thread.name = body.name
    return {"status": "ok", "thread_id": thread_id, "name": thread.name}


# ── Delete thread ─────────────────────────────────────────────────────────────

@router.delete("/{session_id}/threads/{thread_id}", summary="Delete a chat thread")
async def delete_thread(session_id: str, thread_id: str) -> dict:
    s = await _get_or_404(session_id)
    if thread_id not in s.threads:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    del s.threads[thread_id]
    return {"status": "deleted", "thread_id": thread_id}


# ── Thread message history ────────────────────────────────────────────────────

@router.get(
    "/{session_id}/threads/{thread_id}/messages",
    response_model=list[MessageOut],
    summary="Full message history for a thread (used to restore chat on page reload)",
)
async def get_thread_messages(session_id: str, thread_id: str) -> list[MessageOut]:
    s = await _get_or_404(session_id)
    thread = s.threads.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return [
        MessageOut(
            message_id=m.message_id,
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            feedback=m.feedback,
            citations=m.citations,
        )
        for m in thread.messages
    ]
