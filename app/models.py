from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class DocumentType(str, Enum):
    pdf = "pdf"
    image = "image"
    text = "text"


class DocumentInfo(BaseModel):
    name: str
    doc_type: DocumentType
    page_count: int
    chunk_count: int
    uploaded_at: datetime


# ── Bike identity ─────────────────────────────────────────────────────────────

class BikeInfo(BaseModel):
    name: Optional[str] = None   # e.g. "Royal Enfield Classic 350"
    make: Optional[str] = None   # e.g. "Royal Enfield"
    model: Optional[str] = None  # e.g. "Classic 350"
    year: Optional[int] = None


class BikeUpdateRequest(BaseModel):
    name: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    document_name: str
    chunks_created: int
    pages_processed: int
    bike_name: Optional[str] = None   # null if detection failed
    sections: List[str] = []          # unique section headings detected


# ── Chat ──────────────────────────────────────────────────────────────────────

class Confidence(str, Enum):
    direct_match = "direct_match"    # top similarity >= threshold
    best_match = "best_match"        # lower similarity, still best available


class ChatRequest(BaseModel):
    session_id: str
    message: str
    thread_id: Optional[str] = None  # omit to create a new thread
    image_base64: Optional[str] = None
    image_media_type: Optional[str] = None  # "image/jpeg" | "image/png" | …


class Citation(BaseModel):
    page_number: int
    section: str
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    thread_id: str
    message_id: str
    answer: str
    citations: List[Citation]
    confidence: Confidence


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackSignal(str, Enum):
    up = "up"
    down = "down"


class FeedbackRequest(BaseModel):
    signal: FeedbackSignal


# ── Threads ───────────────────────────────────────────────────────────────────

class MessageOut(BaseModel):
    message_id: str
    role: str          # "user" | "assistant"
    content: str
    timestamp: datetime
    feedback: Optional[str] = None
    citations: List[Citation] = []


class ThreadOut(BaseModel):
    thread_id: str
    name: str
    created_at: datetime
    message_count: int


class ThreadRenameRequest(BaseModel):
    name: str


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    bike_info: BikeInfo
    document_count: int
    thread_count: int
    total_chunks: int


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    bike_info: BikeInfo
    documents: List[DocumentInfo]
    sections: List[str]
    total_chunks: int
    total_queries: int
    threads: List[ThreadOut]
