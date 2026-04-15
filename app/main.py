from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, session, upload

app = FastAPI(
    title="GatiUstaad",
    description=(
        "AI-powered bike manual troubleshooting bot. "
        "Upload an owner's/service manual (PDF, image, or text), "
        "then ask questions — answers are grounded strictly in the manual "
        "with page-level citations."
    ),
    version="1.2.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(session.router, tags=["Sessions"])


@app.get("/health", tags=["Health"], summary="Liveness check")
async def health() -> dict:
    return {"status": "ok", "service": "GatiUstaad", "version": "1.2.0"}
