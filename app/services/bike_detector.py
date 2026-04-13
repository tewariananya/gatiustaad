"""
Infers bike make / model / year from the first few chunks of an uploaded manual.

Uses Claude with a tightly scoped prompt; falls back to an empty BikeInfo
on any parse or API error so it never blocks the upload flow.
"""
import json
import logging
from typing import List

from anthropic import AsyncAnthropic

from app.config import settings
from app.models import BikeInfo
from app.session_store import ChunkRecord

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract vehicle information from manual text. "
    "Respond ONLY with a single valid JSON object — no markdown, no prose."
)

_PROMPT_TMPL = """\
Read the following excerpt from a vehicle owner's or service manual and extract:
- make   : manufacturer name (e.g. "Royal Enfield", "Honda", "TVS")
- model  : model name (e.g. "Classic 350", "Activa 6G", "Apache RTR 160")
- year   : model year as an integer (e.g. 2022), or null if not found
- name   : a friendly combined label (e.g. "Royal Enfield Classic 350 (2022)")

Return JSON with exactly these four keys. Set a key to null if the value cannot
be determined from the text.

MANUAL EXCERPT:
{context}"""


async def detect_bike(chunks: List[ChunkRecord], client: AsyncAnthropic) -> BikeInfo:
    """Return BikeInfo extracted from the first 5 chunks; returns empty BikeInfo on failure."""
    if not chunks:
        return BikeInfo()

    context = "\n\n---\n\n".join(c.text for c in chunks[:5])
    prompt = _PROMPT_TMPL.format(context=context[:6000])  # hard cap to avoid huge tokens

    try:
        response = await client.messages.create(
            model=settings.claude_model,
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data: dict = json.loads(raw)

        make = data.get("make") or None
        model = data.get("model") or None
        year_raw = data.get("year")
        year = int(year_raw) if year_raw else None

        # Build a fallback friendly name if "name" key is absent / empty
        name = data.get("name") or None
        if not name and (make or model):
            parts = [p for p in [make, model, f"({year})" if year else None] if p]
            name = " ".join(parts)

        return BikeInfo(name=name, make=make, model=model, year=year)

    except Exception as exc:
        logger.warning("Bike detection failed: %s", exc)
        return BikeInfo()
