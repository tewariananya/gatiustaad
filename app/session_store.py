import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models import BikeInfo, Citation, DocumentInfo


# ── Chunk record ──────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    chunk_id: int
    document_name: str
    page_number: int
    section: str
    text: str
    token_count: int


# ── Chat thread / message storage ─────────────────────────────────────────────

@dataclass
class ChatMessage:
    message_id: str
    role: str                          # "user" | "assistant"
    content: str
    timestamp: datetime
    feedback: Optional[str] = None     # "up" | "down" | None
    citations: List[Citation] = field(default_factory=list)


@dataclass
class ChatThread:
    thread_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = field(default_factory=list)


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    session_id: str
    created_at: datetime
    bike_info: BikeInfo = field(default_factory=BikeInfo)
    documents: List[DocumentInfo] = field(default_factory=list)
    chunks: List[ChunkRecord] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    manual_text: str = ""
    bm25: Any = None                   # BM25Okapi instance, rebuilt on each upload
    threads: Dict[str, ChatThread] = field(default_factory=dict)
    query_count: int = 0

    def get_or_create_thread(self, thread_id: Optional[str]) -> ChatThread:
        if thread_id and thread_id in self.threads:
            return self.threads[thread_id]
        tid = thread_id or str(uuid.uuid4())
        thread = ChatThread(thread_id=tid, name="New chat", created_at=datetime.utcnow())
        self.threads[tid] = thread
        return thread


# ── Store singleton ───────────────────────────────────────────────────────────

class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        async with self._lock:
            self._sessions[session_id] = SessionData(
                session_id=session_id,
                created_at=datetime.utcnow(),
            )
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    async def get_or_create(self, session_id: Optional[str]) -> tuple[SessionData, bool]:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id], False
        new_id = session_id or str(uuid.uuid4())
        async with self._lock:
            if new_id not in self._sessions:
                self._sessions[new_id] = SessionData(
                    session_id=new_id,
                    created_at=datetime.utcnow(),
                )
        return self._sessions[new_id], True

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            return True

    def all_sessions(self) -> List[SessionData]:
        return list(self._sessions.values())


session_store = SessionStore()
