"""Low-overhead observability writes: HTTP requests, errors, LLM/search calls, feature usage.

Rows go onto an in-process queue and a daemon thread batch-inserts them, so a
request never waits on the database to record itself. If the database is down,
events are dropped after logging; observability must never break the app.
"""

import json
import queue
import threading
import time
import traceback
from contextvars import ContextVar

from loguru import logger
from sqlalchemy import text

from db import get_engine

current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)

_TABLE_SQL = {
    "api_requests": "INSERT INTO api_requests (request_id, method, route, status_code, duration_ms, client_id, user_agent_family) "
                    "VALUES (:request_id, :method, :route, :status_code, :duration_ms, :client_id, :user_agent_family)",
    "error_events": "INSERT INTO error_events (request_id, source, route, error_type, message, stack) "
                    "VALUES (:request_id, :source, :route, :error_type, :message, :stack)",
    "llm_calls": "INSERT INTO llm_calls (request_id, provider, purpose, model, latency_ms, prompt_tokens, completion_tokens, success, error) "
                 "VALUES (:request_id, :provider, :purpose, :model, :latency_ms, :prompt_tokens, :completion_tokens, :success, :error)",
    "feature_events": "INSERT INTO feature_events (client_id, feature, detail) VALUES (:client_id, :feature, CAST(:detail AS JSONB))",
}

_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue(maxsize=10_000)
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_enabled = True


def disable() -> None:
    """Used by tests so nothing is written."""
    global _enabled
    _enabled = False


def _ensure_worker() -> None:
    global _worker
    if _worker and _worker.is_alive():
        return
    with _worker_lock:
        if _worker and _worker.is_alive():
            return
        _worker = threading.Thread(target=_run, name="observability-writer", daemon=True)
        _worker.start()


def _run() -> None:
    while True:
        batch = [_queue.get()]
        deadline = time.monotonic() + 1.0
        while len(batch) < 200:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(_queue.get(timeout=remaining))
            except queue.Empty:
                break
        flush(batch)


def flush(batch: list[tuple[str, dict]]) -> None:
    by_table: dict[str, list[dict]] = {}
    for table, row in batch:
        by_table.setdefault(table, []).append(row)
    try:
        with get_engine().begin() as conn:
            for table, rows in by_table.items():
                conn.execute(text(_TABLE_SQL[table]), rows)
    except Exception as exc:  # never let observability take the app down
        logger.warning("observability flush dropped {} rows: {}", len(batch), exc)


def emit(table: str, **row) -> None:
    if not _enabled:
        return
    try:
        _queue.put_nowait((table, row))
        _ensure_worker()
    except queue.Full:
        logger.warning("observability queue full, dropping {} event", table)


def drain(timeout: float = 5.0) -> None:
    """Flush everything queued right now (used by CLI jobs before exit)."""
    items = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            items.append(_queue.get_nowait())
        except queue.Empty:
            break
    if items:
        flush(items)


def record_error(source: str, exc: BaseException | str, route: str | None = None) -> None:
    if isinstance(exc, BaseException):
        error_type = type(exc).__name__
        message = str(exc)[:2000]
        stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]
    else:
        error_type, message, stack = "message", str(exc)[:2000], None
    emit("error_events", request_id=current_request_id.get(), source=source, route=route,
         error_type=error_type, message=message, stack=stack)


def record_feature(feature: str, client_id: str | None = None, **detail) -> None:
    emit("feature_events", client_id=client_id, feature=feature, detail=json.dumps(detail, default=str))


def groq_chat(client, purpose: str, **kwargs):
    """client.chat.completions.create with latency, token and failure capture."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        emit("llm_calls", request_id=current_request_id.get(), provider="groq", purpose=purpose,
             model=kwargs.get("model"), latency_ms=round((time.perf_counter() - start) * 1000, 2),
             prompt_tokens=None, completion_tokens=None, success=False, error=str(exc)[:1000])
        record_error("groq", exc)
        raise
    usage = getattr(response, "usage", None)
    emit("llm_calls", request_id=current_request_id.get(), provider="groq", purpose=purpose,
         model=kwargs.get("model"), latency_ms=round((time.perf_counter() - start) * 1000, 2),
         prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None),
         success=True, error=None)
    return response


def tavily_search(client, purpose: str, **kwargs):
    start = time.perf_counter()
    try:
        response = client.search(**kwargs)
    except Exception as exc:
        emit("llm_calls", request_id=current_request_id.get(), provider="tavily", purpose=purpose, model=None,
             latency_ms=round((time.perf_counter() - start) * 1000, 2), prompt_tokens=None, completion_tokens=None,
             success=False, error=str(exc)[:1000])
        record_error("tavily", exc)
        raise
    emit("llm_calls", request_id=current_request_id.get(), provider="tavily", purpose=purpose, model=None,
         latency_ms=round((time.perf_counter() - start) * 1000, 2), prompt_tokens=None, completion_tokens=None,
         success=True, error=None)
    return response


def _ua_family(ua: str) -> str:
    ua = (ua or "").lower()
    if not ua:
        return "unknown"
    if any(b in ua for b in ("bot", "spider", "crawl", "curl", "python", "httpx")):
        return "bot"
    if any(m in ua for m in ("mobile", "android", "iphone")):
        return "mobile"
    return "desktop"


def install_http_middleware(app) -> None:
    """Record every request (route template, status, latency) and every unhandled exception."""
    import uuid

    @app.middleware("http")
    async def _observe(request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = current_request_id.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-Id"] = rid
            return response
        except Exception as exc:
            record_error("http", exc, route=request.url.path)
            raise
        finally:
            route = request.scope.get("route")
            template = getattr(route, "path", None) or request.url.path
            emit("api_requests", request_id=rid, method=request.method, route=template, status_code=status,
                 duration_ms=round((time.perf_counter() - start) * 1000, 2),
                 client_id=(request.headers.get("x-client-id") or None),
                 user_agent_family=_ua_family(request.headers.get("user-agent", "")))
            current_request_id.reset(token)
