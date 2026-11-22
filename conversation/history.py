"""Chat history: save each question and answer to its thread, list and read threads per device.

A thread belongs to the anonymous client id that created it (a random UUID the
browser keeps). Another device can never read or write it. Only what the user
typed and what the app answered is stored; no IP address or personal data.
"""

import json
import uuid

from loguru import logger
from sqlalchemy import text

from db import get_engine

MAX_THREADS_PER_CLIENT = 200
MAX_LISTINGS_STORED = 9
TITLE_LENGTH = 80


def valid_uuid(value: str | None) -> str | None:
    try:
        return str(uuid.UUID(value or ""))
    except ValueError:
        return None


def _title(question: str) -> str:
    q = " ".join((question or "").split())
    return q if len(q) <= TITLE_LENGTH else q[: TITLE_LENGTH - 1].rstrip() + "…"


def history_payload(payload: dict) -> dict:
    """The parts of an API answer worth replaying later: the answer card, top listings, or advice."""
    listings = list(payload.get("results") or [])
    for group in payload.get("areas") or []:
        listings += group.get("results") or []
    for option in payload.get("options") or []:
        listings += option.get("results") or []
    seen, unique = set(), []
    for item in listings:
        url = item.get("url") if isinstance(item, dict) else None
        if url and url not in seen:
            seen.add(url)
            unique.append(item)
    out = {"answer": payload.get("answer"), "listings": unique[:MAX_LISTINGS_STORED]}
    if payload.get("query_type") == "advice":
        out["advice"] = {"text": (payload.get("answer") or {}).get("message", ""), "sources": payload.get("sources") or []}
    return out


def save_exchange(client_id: str | None, thread_id: str | None, question: str, payload: dict, engine=None) -> bool:
    """Append one question and answer to a thread, creating the thread on first use.

    Returns False (and saves nothing) when ids are missing or the thread belongs to another device.
    """
    cid, tid = valid_uuid(client_id), valid_uuid(thread_id)
    if not cid or not tid or not (question or "").strip():
        return False
    eng = engine or get_engine()
    stored = json.dumps(history_payload(payload), default=str)
    request_id = ((payload.get("answer") or {}).get("request_id")) if isinstance(payload, dict) else None
    with eng.begin() as conn:
        owner = conn.execute(text("SELECT client_id FROM chat_threads WHERE id = :t FOR UPDATE"), {"t": tid}).scalar()
        if owner is None:
            count = conn.execute(text("SELECT COUNT(*) FROM chat_threads WHERE client_id = :c"), {"c": cid}).scalar() or 0
            if count >= MAX_THREADS_PER_CLIENT:
                # Keep history bounded: drop this device's oldest thread.
                conn.execute(text("""DELETE FROM chat_threads WHERE id = (
                    SELECT id FROM chat_threads WHERE client_id = :c ORDER BY updated_at ASC LIMIT 1)"""), {"c": cid})
            conn.execute(text("INSERT INTO chat_threads (id, client_id, title) VALUES (:t, :c, :title)"),
                         {"t": tid, "c": cid, "title": _title(question)})
        elif owner != cid:
            logger.warning("Refusing to write to a chat thread owned by another device")
            return False
        conn.execute(text("""INSERT INTO chat_messages (thread_id, question, payload, request_id)
                             VALUES (:t, :q, CAST(:p AS JSONB), :r)"""),
                     {"t": tid, "q": question.strip()[:4000], "p": stored, "r": request_id})
        conn.execute(text("UPDATE chat_threads SET message_count = message_count + 1, updated_at = now() WHERE id = :t"),
                     {"t": tid})
    return True


def list_threads(client_id: str, limit: int = 50, engine=None) -> list[dict]:
    eng = engine or get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.id, t.title, t.message_count, t.created_at, t.updated_at,
                   (SELECT m.question FROM chat_messages m WHERE m.thread_id = t.id ORDER BY m.id DESC LIMIT 1) AS last_question
            FROM chat_threads t WHERE t.client_id = :c ORDER BY t.updated_at DESC LIMIT :lim
        """), {"c": client_id, "lim": limit}).mappings().all()
    return [dict(r) for r in rows]


def get_thread(client_id: str, thread_id: str, engine=None) -> dict | None:
    eng = engine or get_engine()
    with eng.connect() as conn:
        thread = conn.execute(text("SELECT id, title, message_count, created_at, updated_at FROM chat_threads "
                                   "WHERE id = :t AND client_id = :c"), {"t": thread_id, "c": client_id}).mappings().first()
        if not thread:
            return None
        messages = conn.execute(text("SELECT id, question, payload, created_at FROM chat_messages "
                                     "WHERE thread_id = :t ORDER BY id"), {"t": thread_id}).mappings().all()
    return {**dict(thread), "messages": [dict(m) for m in messages]}


def delete_thread(client_id: str, thread_id: str, engine=None) -> bool:
    eng = engine or get_engine()
    with eng.begin() as conn:
        n = conn.execute(text("DELETE FROM chat_threads WHERE id = :t AND client_id = :c"),
                         {"t": thread_id, "c": client_id}).rowcount
        conn.execute(text("DELETE FROM conversation_sessions WHERE session_id = :t"), {"t": thread_id})
    return bool(n)


def delete_all_threads(client_id: str, engine=None) -> int:
    eng = engine or get_engine()
    with eng.begin() as conn:
        ids = conn.execute(text("DELETE FROM chat_threads WHERE client_id = :c RETURNING id"), {"c": client_id}).scalars().all()
        if ids:
            conn.execute(text("DELETE FROM conversation_sessions WHERE session_id = ANY(:ids)"), {"ids": list(ids)})
    return len(ids)
