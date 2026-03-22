"""
MongoDB Atlas Service
Logs obstacle sessions and query history for persistence and recall.

Connection matches repo ``test.py`` (reference):

  - ``MONGODB_URI`` — default ``mongodb+srv://admin:elsee@cluster0.uuiqxb1.mongodb.net/``
  - ``MONGODB_DB_NAME`` — default ``seefore``

STT transcripts from the mobile app (``POST /api/transcripts``) are stored in the **same**
database and collection as the reference script: ``{MONGODB_DB_NAME}.test_collection``.
Documents use BSON UTC ``created_at`` (timezone-aware) for range queries and sorting.

Ask Q&A review logs use ``ask_conversations`` in the same database.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient

log = logging.getLogger("seefore.mongodb")


def utc_now() -> datetime:
    """UTC instant as timezone-aware datetime (BSON Date — good for ``$gte`` / ``$lte`` / sort)."""
    return datetime.now(timezone.utc)


# Same env keys and defaults as ``test.py`` at repo root
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://admin:elsee@cluster0.uuiqxb1.mongodb.net/",
).strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "seefore").strip()

# STT transcripts: same collection name as ``test.py`` (``db["test_collection"]``)
AUDIO_TRANSCRIPTS_COLLECTION = "test_collection"
ASK_CONVERSATIONS_COLLECTION = "test_collection"


def atlas_cluster_host(uri: str) -> str:
    """Return Atlas hostname from URI (no credentials)."""
    if not uri or "@" not in uri:
        return ""
    try:
        host = uri.split("@", 1)[1]
        return host.split("/")[0].split("?")[0].strip()
    except Exception:
        return ""


def transcript_storage_location() -> dict[str, Any]:
    """Where STT transcript documents are written (same as ``test.py`` collection)."""
    return {
        "atlas_cluster_host": atlas_cluster_host(MONGODB_URI),
        "database": MONGODB_DB_NAME,
        "collection": AUDIO_TRANSCRIPTS_COLLECTION,
    }


def ask_conversation_storage_location() -> dict[str, Any]:
    """Where full Ask Q&A turns are stored."""
    return {
        "atlas_cluster_host": atlas_cluster_host(MONGODB_URI),
        "database": MONGODB_DB_NAME,
        "collection": ASK_CONVERSATIONS_COLLECTION,
    }


_client: Optional[AsyncIOMotorClient] = None
_db = None


async def connect_mongo():
    global _client, _db
    if not MONGODB_URI:
        log.warning("MONGODB_URI not set — MongoDB logging disabled")
        return
    try:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[MONGODB_DB_NAME]
        await _client.admin.command("ping")
        loc = transcript_storage_location()
        log.info(
            "✅ MongoDB Atlas connected — cluster host=%s database=%s (STT transcripts → %s)",
            loc["atlas_cluster_host"] or "(from MONGODB_URI)",
            loc["database"],
            loc["collection"],
        )
        print("Collection: ",loc["database"], loc["collection"])
    except Exception as e:
        log.error(f"MongoDB connection failed: {e}")
        _client = None
        _db = None


async def close_mongo():
    global _client
    if _client:
        _client.close()
        log.info("MongoDB connection closed")


async def log_obstacle_session(data: dict):
    """Log obstacle detection frame to MongoDB."""
    if _db is None:
        return
    try:
        data["created_at"] = utc_now()
        # await _db["obstacle_sessions"].insert_one(data)
    except Exception as e:
        log.error(f"MongoDB obstacle log error: {e}")


async def log_query_session(data: dict):
    """Log user query + Gemini response to MongoDB."""
    if _db is None:
        return
    try:
        data["created_at"] = utc_now()
        # await _db["query_sessions"].insert_one(data)
    except Exception as e:
        log.error(f"MongoDB query log error: {e}")


async def log_audio_transcript(
    text: str,
    source: str = "mobile_ask",
    platform: Optional[str] = None,
) -> bool:
    """Insert one STT transcript into ``{MONGODB_DB_NAME}.test_collection`` (same as ``test.py``)."""
    if _db is None:
        return False
    if not (text or "").strip():
        return False
    doc: dict[str, Any] = {
        "transcript": text.strip(),
        "source": source,
        "created_at": utc_now(),
    }
    if platform:
        doc["platform"] = platform
    try:
        await _db[AUDIO_TRANSCRIPTS_COLLECTION].insert_one(doc)
        loc = transcript_storage_location()
        log.info(
            "Saved transcript to Atlas host=%s db=%s collection=%s",
            loc["atlas_cluster_host"],
            loc["database"],
            loc["collection"],
        )
        return True
    except Exception as e:
        log.error(f"MongoDB transcript log error ({AUDIO_TRANSCRIPTS_COLLECTION}): {e}")
        return False


async def log_ask_conversation(
    query: str,
    response: str,
    source: str = "mobile_ask",
    platform: Optional[str] = None,
) -> bool:
    """Store one Ask turn (user text + model reply) for later review."""
    if _db is None:
        return False
    q, r = (query or "").strip(), (response or "").strip()
    if not q and not r:
        return False
    doc: dict[str, Any] = {
        "query": q,
        "response": r,
        "source": source,
        "created_at": utc_now(),
    }
    if platform:
        doc["platform"] = platform
    try:
        await _db[ASK_CONVERSATIONS_COLLECTION].insert_one(doc)
        loc = ask_conversation_storage_location()
        log.info(
            "Saved ask conversation to Atlas host=%s db=%s collection=%s",
            loc["atlas_cluster_host"],
            loc["database"],
            loc["collection"],
        )
        return True
    except Exception as e:
        log.error(f"MongoDB ask_conversations log error: {e}")
        return False


async def get_app_config() -> dict:
    """
    Single document in `app_config` with _id 'default' — drives mobile public config (URLs only).
    """
    if _db is None:
        return {}
    try:
        doc = await _db["app_config"].find_one({"_id": "default"})
        if not doc:
            return {}
        out = dict(doc)
        out.pop("_id", None)
        return out
    except Exception as e:
        log.error(f"MongoDB app_config read error: {e}")
        return {}


async def get_recent_sessions(user_id: str, limit: int = 20) -> list:
    """Retrieve recent sessions for a user."""
    if _db is None:
        return []
    try:
        cursor = _db["query_sessions"].find(
            {"user_id": user_id},
            sort=[("created_at", -1)],
            limit=limit
        )
        docs = await cursor.to_list(length=limit)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return docs
    except Exception as e:
        log.error(f"MongoDB fetch error: {e}")
        return []
