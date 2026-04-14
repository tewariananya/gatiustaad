"""
POST /upload
Accepts PDF, image, or raw text. Parses, chunks, embeds (locally), and stores.
Returns session_id, detected bike_name, and extracted section list.
"""
from datetime import datetime
from typing import Optional

from anthropic import AsyncAnthropic
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models import DocumentInfo, DocumentType, UploadResponse
from app.services.bike_detector import detect_bike
from app.services.chunker import chunk_pages
from app.services.document_processor import (
    extract_image_text,
    extract_pdf_pages,
    extract_raw_text,
)
from app.services.vector_store import add_chunks
from app.session_store import session_store

router = APIRouter()

_IMAGE_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _guess_image_mime(filename: str, content_type: str) -> Optional[str]:
    if content_type in _IMAGE_MIME:
        return content_type
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return "image/jpeg" if ext in _IMAGE_EXT else None


@router.post("/upload", response_model=UploadResponse, summary="Upload a bike manual")
async def upload_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
) -> UploadResponse:
    if file is None and not text:
        raise HTTPException(status_code=400, detail="Provide either a `file` or `text` field.")

    session, _ = await session_store.get_or_create(session_id)
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # ── Extract pages ─────────────────────────────────────────────────────────
    if file is not None:
        raw = await file.read()
        filename: str = file.filename or "document"
        ct: str = file.content_type or ""

        if ct == "application/pdf" or filename.lower().endswith(".pdf"):
            pages = extract_pdf_pages(raw)
            doc_type = DocumentType.pdf
        elif (mime := _guess_image_mime(filename, ct)) is not None:
            pages = await extract_image_text(raw, mime, anthropic_client)
            doc_type = DocumentType.image
        else:
            try:
                pages = extract_raw_text(raw.decode("utf-8"))
                doc_type = DocumentType.text
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type '{ct}'. Upload a PDF, image, or plain-text file.",
                )
    else:
        pages = extract_raw_text(text)  # type: ignore[arg-type]
        filename = "inline_text"
        doc_type = DocumentType.text

    if not pages:
        raise HTTPException(status_code=422, detail="No text could be extracted.")

    # ── Chunk → embed (local) → store ─────────────────────────────────────────
    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks.")

    chunk_count = await add_chunks(session, chunks, filename)

    # ── Collect sections + manual text ────────────────────────────────────────
    for p in pages:
        if p.section and p.section not in session.sections:
            session.sections.append(p.section)
    session.manual_text += "\n\n".join(p.text for p in pages) + "\n\n"

    session.documents.append(DocumentInfo(
        name=filename,
        doc_type=doc_type,
        page_count=len(pages),
        chunk_count=chunk_count,
        uploaded_at=datetime.utcnow(),
    ))

    # ── Bike detection (best-effort) ──────────────────────────────────────────
    if session.bike_info.name is None:
        bike_info = await detect_bike(session.chunks, anthropic_client)
        if bike_info.name:
            session.bike_info = bike_info

    return UploadResponse(
        session_id=session.session_id,
        document_name=filename,
        chunks_created=chunk_count,
        pages_processed=len(pages),
        bike_name=session.bike_info.name,
        sections=session.sections,
    )
