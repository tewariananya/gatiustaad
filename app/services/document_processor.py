"""
Extracts text (with page/section metadata) from PDFs, images, and raw text.

PDF  → PyMuPDF page-by-page extraction
Image → Claude Vision OCR (returns a single synthetic "page 1")
Text  → wrapped directly
"""
import base64
import re
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from anthropic import AsyncAnthropic

from app.config import settings


@dataclass
class PageContent:
    page_num: int   # 1-indexed
    text: str
    section: str    # first detectable heading, or ""


# ── Section detection ─────────────────────────────────────────────────────────

_NUMBERED_SECTION = re.compile(r"^(\d+\.)+\d*\s+\w")
_CHAPTER_SECTION = re.compile(r"^(chapter|section)\s+\d+", re.IGNORECASE)


def _detect_section(text: str) -> str:
    """Return the first heading-like line found in the first 5 non-empty lines."""
    for line in text.strip().splitlines()[:5]:
        line = line.strip()
        if not line:
            continue
        if _NUMBERED_SECTION.match(line) or _CHAPTER_SECTION.match(line):
            return line[:120]
        # ALL-CAPS short line → likely a heading
        if line.isupper() and 4 <= len(line) <= 80:
            return line[:120]
    return ""


# ── PDF ───────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_bytes: bytes) -> List[PageContent]:
    """Return one PageContent per non-empty page."""
    pages: List[PageContent] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(PageContent(
                    page_num=i + 1,
                    text=text,
                    section=_detect_section(text),
                ))
    finally:
        doc.close()
    return pages


# ── Image (Claude Vision OCR) ─────────────────────────────────────────────────

async def extract_image_text(
    image_bytes: bytes,
    media_type: str,
    client: AsyncAnthropic,
) -> List[PageContent]:
    """Use Claude Vision to OCR a standalone image and return it as page 1."""
    image_b64 = base64.standard_b64encode(image_bytes).decode()

    response = await client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Extract ALL text from this image exactly as it appears. "
                        "Preserve headings, section numbers, lists, and any structured content. "
                        "Output only the extracted text — no commentary."
                    ),
                },
            ],
        }],
    )

    text = response.content[0].text
    return [PageContent(page_num=1, text=text, section=_detect_section(text))]


# ── Raw text ──────────────────────────────────────────────────────────────────

def extract_raw_text(text: str) -> List[PageContent]:
    return [PageContent(page_num=1, text=text, section=_detect_section(text))]
