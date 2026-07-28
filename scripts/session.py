import json
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import text

from db import get_engine
from entities import PropertyEntities

SESSION_TTL_HOURS = 6


def load_session_entities(session_id: str) -> Optional[PropertyEntities]:
    if not session_id:
        return None

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT last_entities, updated_at
                FROM conversation_sessions
                WHERE session_id = :session_id
            """),
            {"session_id": session_id},
        ).mappings().first()

    if row is None:
        return None

    age_hours = (datetime.now(timezone.utc) - row["updated_at"]).total_seconds() / 3600
    if age_hours > SESSION_TTL_HOURS:
        logger.info("Session {} expired ({:.1f}h old), ignoring stored entities", session_id, age_hours)
        return None

    try:
        return PropertyEntities.model_validate(row["last_entities"])
    except Exception as exc:
        logger.warning("Session {} has unparseable stored entities, ignoring: {}", session_id, exc)
        return None


def save_session_entities(session_id: str, entities: PropertyEntities) -> None:
    if not session_id:
        return

    engine = get_engine()
    payload = json.dumps(entities.model_dump(mode="json"))
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_sessions (session_id, last_entities, updated_at)
                VALUES (:session_id, :last_entities, now())
                ON CONFLICT (session_id)
                DO UPDATE SET last_entities = :last_entities, updated_at = now()
            """),
            {"session_id": session_id, "last_entities": payload},
        )
    logger.info("Saved session {} entities: query_type={}", session_id, entities.query_type)