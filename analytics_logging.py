from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from db import get_engine

MIGRATIONS_DIR = Path("sql/migrations")


def apply_migrations(engine=None) -> list[str]:
    eng = engine or get_engine()
    applied = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        with eng.begin() as conn:
            conn.execute(text(sql))
        applied.append(path.name)
    logger.info("Applied migrations: {}", applied)
    return applied


def log_question(request_id, session_id=None, raw_question="", parsed_location=None, parsed_property_type=None, status="error", listing_count=0, confidence=None, latency_ms=0.0, time_to_first_event_ms=None, error_code=None) -> int:
    engine = get_engine()
    # No IP or personal data is stored by design.
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """INSERT INTO question_events (request_id, session_id, raw_question, parsed_location, parsed_property_type, status, listing_count, confidence, latency_ms, time_to_first_event_ms, error_code)
                VALUES (:request_id, :session_id, :raw_question, :parsed_location, :parsed_property_type, :status, :listing_count, :confidence, :latency_ms, :ttfe, :error_code)
                RETURNING id"""
            ),
            {
                "request_id": str(request_id),
                "session_id": session_id,
                "raw_question": (raw_question or "")[:4000],
                "parsed_location": parsed_location,
                "parsed_property_type": parsed_property_type,
                "status": status,
                "listing_count": int(listing_count or 0),
                "confidence": confidence,
                "latency_ms": float(latency_ms or 0.0),
                "ttfe": float(time_to_first_event_ms) if time_to_first_event_ms is not None else None,
                "error_code": error_code,
            },
        ).mappings().first()
    return int(row["id"]) if row else 0


def record_scraper_run(triggered_by="schedule", status="queued", discovered=0, added=0, updated=0, duplicates_skipped=0, validation_failures=0, error_count=0, log_excerpt="", completed=False) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """INSERT INTO scraper_runs (triggered_by, started_at, completed_at, status, listings_discovered, listings_added, listings_updated, duplicates_skipped, validation_failures, error_count, log_excerpt)
                VALUES (:triggered_by, now(), :completed_at, :status, :discovered, :added, :updated, :duplicates_skipped, :validation_failures, :error_count, :log_excerpt)
                RETURNING id"""
            ),
            {
                "triggered_by": triggered_by,
                "completed_at": datetime.now(UTC) if completed or status in ("success", "failed") else None,
                "status": status,
                "discovered": int(discovered),
                "added": int(added),
                "updated": int(updated),
                "duplicates_skipped": int(duplicates_skipped),
                "validation_failures": int(validation_failures),
                "error_count": int(error_count),
                "log_excerpt": (log_excerpt or "")[:8000],
            },
        ).mappings().first()
    return int(row["id"]) if row else 0
