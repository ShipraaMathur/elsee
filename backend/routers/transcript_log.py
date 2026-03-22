"""
POST /api/transcripts — store Gemini STT text from the mobile app in MongoDB Atlas.

Persists STT text to ``MONGODB_DB_NAME.test_collection`` (same DB/collection as repo ``test.py``),
using ``MONGODB_URI`` (e.g. ``cluster0.uuiqxb1.mongodb.net``). Ask Q&A goes to ``ask_conversations``.

Optional header ``X-Transcript-Secret`` when ``TRANSCRIPT_LOG_SECRET`` is set on the server.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from models.schemas import (
    AskConversationLogRequest,
    AskConversationLogResponse,
    TranscriptLogRequest,
    TranscriptLogResponse,
)
from services.mongodb import (
    ask_conversation_storage_location,
    log_ask_conversation,
    log_audio_transcript,
    transcript_storage_location,
)

router = APIRouter()

_SECRET = os.getenv("TRANSCRIPT_LOG_SECRET", "").strip()


@router.post("/transcripts", response_model=TranscriptLogResponse)
async def save_transcript(
    body: TranscriptLogRequest,
    x_transcript_secret: Optional[str] = Header(None, alias="X-Transcript-Secret"),
):
    if _SECRET and x_transcript_secret != _SECRET:
        raise HTTPException(status_code=403, detail="Invalid X-Transcript-Secret")

    loc = transcript_storage_location()
    stored = await log_audio_transcript(
        body.transcript,
        body.source or "mobile_ask",
        platform=body.platform,
    )
    return TranscriptLogResponse(
        ok=True,
        stored=stored,
        atlas_cluster_host=loc["atlas_cluster_host"] or None,
        database=loc["database"],
        collection=loc["collection"],
    )


@router.post("/ask-conversations", response_model=AskConversationLogResponse)
async def save_ask_conversation(
    body: AskConversationLogRequest,
    x_transcript_secret: Optional[str] = Header(None, alias="X-Transcript-Secret"),
):
    """Store one Ask Q&A turn (device Gemini) for review in ``ask_conversations``."""
    if _SECRET and x_transcript_secret != _SECRET:
        raise HTTPException(status_code=403, detail="Invalid X-Transcript-Secret")

    loc = ask_conversation_storage_location()
    stored = await log_ask_conversation(
        body.query,
        body.response,
        body.source or "mobile_ask",
        platform=body.platform,
    )
    return AskConversationLogResponse(
        ok=True,
        stored=stored,
        atlas_cluster_host=loc["atlas_cluster_host"] or None,
        database=loc["database"],
        collection=loc["collection"],
    )
