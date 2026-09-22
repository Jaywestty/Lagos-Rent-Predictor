import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from loguru import logger
from sqlalchemy import text

from db import get_engine

_current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)
_call_counts: ContextVar[dict] = ContextVar("call_counts", default=None)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def start_trace() -> str:
    trace_id = new_trace_id()
    _current_trace_id.set(trace_id)
    _call_counts.set({})
    return trace_id


def get_current_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def record_call(service: str) -> None:
    if _current_trace_id.get() is None:
        return
    counts = dict(_call_counts.get() or {})
    counts[service] = counts.get(service, 0) + 1
    _call_counts.set(counts)


def _save_stage_trace(trace_id, stage, query_type, duration_ms, call_counts, success, error_text):
    if trace_id is None:
        logger.warning("No active trace_id, skipping trace persistence for stage={}", stage)
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO traces (trace_id, stage, query_type, duration_ms, call_counts, success, error, created_at)
                VALUES (:trace_id, :stage, :query_type, :duration_ms, :call_counts, :success, :error, now())
            """),
            {
                "trace_id": trace_id,
                "stage": stage,
                "query_type": query_type,
                "duration_ms": duration_ms,
                "call_counts": json.dumps(call_counts),
                "success": success,
                "error": error_text,
            },
        )
    logger.info(
        "Trace stage: trace_id={} stage={} duration_ms={} calls={} success={}",
        trace_id, stage, duration_ms, call_counts, success,
    )


@contextmanager
def trace_stage(stage: str, query_type: Optional[str] = None):
    trace_id = _current_trace_id.get()
    counts_before = dict(_call_counts.get() or {})
    start = time.perf_counter()
    success = True
    error_text = None
    try:
        yield
    except Exception as exc:
        success = False
        error_text = str(exc)
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        counts_after = dict(_call_counts.get() or {})
        stage_calls = {
            service: counts_after.get(service, 0) - counts_before.get(service, 0)
            for service in counts_after
            if counts_after.get(service, 0) - counts_before.get(service, 0) > 0
        }
        _save_stage_trace(trace_id, stage, query_type, duration_ms, stage_calls, success, error_text)