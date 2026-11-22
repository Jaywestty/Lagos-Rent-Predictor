"""Admin observability: traffic, latency, errors, AI/search calls, data health, scraper, feature usage."""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Cookie, HTTPException, Query, Request
from sqlalchemy import text

from admin.auth import get_current_admin, require_csrf
from db import get_engine

router = APIRouter(prefix="/api/admin/observability", tags=["admin-observability"])


def _since(hours: int) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


def _next_weekly_run(now: datetime) -> datetime:
    """Next Monday 01:00 UTC, matching .github/workflows/weekly-scrape.yml."""
    days = (0 - now.weekday()) % 7
    candidate = (now + timedelta(days=days)).replace(hour=1, minute=0, second=0, microsecond=0)
    return candidate if candidate > now else candidate + timedelta(days=7)


def _rows(conn, sql: str, params: dict) -> list[dict]:
    return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


def _f(v):
    return float(v) if v is not None else None


@router.get("/overview")
def overview(request: Request, hours: int = Query(24, ge=1, le=24 * 90), admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    p = {"since": _since(hours)}
    with get_engine().connect() as conn:
        traffic = conn.execute(text("""
            SELECT COUNT(*) AS requests,
                   COUNT(*) FILTER (WHERE status_code >= 500) AS server_errors,
                   COUNT(*) FILTER (WHERE status_code BETWEEN 400 AND 499) AS client_errors,
                   COUNT(DISTINCT client_id) AS unique_clients,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
            FROM api_requests WHERE created_at >= :since AND route NOT LIKE '/api/admin%'
        """), p).mappings().one()
        routes = _rows(conn, """
            SELECT method, route, COUNT(*) AS requests,
                   COUNT(*) FILTER (WHERE status_code >= 500) AS errors,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
            FROM api_requests WHERE created_at >= :since GROUP BY method, route ORDER BY requests DESC LIMIT 25
        """, p)
        devices = _rows(conn, "SELECT user_agent_family AS family, COUNT(*) AS c FROM api_requests WHERE created_at >= :since GROUP BY 1 ORDER BY 2 DESC", p)
        ai = _rows(conn, """
            SELECT provider, purpose, COUNT(*) AS calls, COUNT(*) FILTER (WHERE NOT success) AS failures,
                   AVG(latency_ms) AS avg_ms, percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, COALESCE(SUM(completion_tokens), 0) AS completion_tokens
            FROM llm_calls WHERE created_at >= :since GROUP BY provider, purpose ORDER BY calls DESC
        """, p)
        errors_by_source = _rows(conn, "SELECT source, COUNT(*) AS c FROM error_events WHERE created_at >= :since GROUP BY 1 ORDER BY 2 DESC", p)
        questions = conn.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE status = 'success') AS success,
                   COUNT(*) FILTER (WHERE status = 'low_data') AS low_data,
                   COUNT(*) FILTER (WHERE status = 'no_data') AS no_data,
                   COUNT(*) FILTER (WHERE status = 'error') AS error,
                   COUNT(*) FILTER (WHERE status IN ('clarification', 'out_of_scope')) AS other,
                   AVG(latency_ms) AS avg_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
                   COUNT(DISTINCT session_id) AS sessions
            FROM question_events WHERE created_at >= :since
        """), p).mappings().one()
        unanswered_areas = _rows(conn, """
            SELECT parsed_location AS area, COUNT(*) AS c FROM question_events
            WHERE created_at >= :since AND status IN ('no_data', 'low_data') AND parsed_location IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """, p)
        stages = _rows(conn, """
            SELECT stage, COUNT(*) AS runs, COUNT(*) FILTER (WHERE NOT success) AS failures,
                   AVG(duration_ms) AS avg_ms, percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
            FROM traces WHERE created_at >= :since GROUP BY stage ORDER BY runs DESC
        """, p)
        features = _rows(conn, "SELECT feature, COUNT(*) AS c, COUNT(DISTINCT client_id) AS clients FROM feature_events WHERE created_at >= :since GROUP BY 1 ORDER BY 2 DESC", p)
        data = conn.execute(text("""
            SELECT COUNT(*) FILTER (WHERE is_duplicate_of IS NULL) AS listings,
                   COUNT(*) FILTER (WHERE is_duplicate_of IS NOT NULL) AS duplicates,
                   COUNT(DISTINCT area) AS areas,
                   COUNT(*) FILTER (WHERE price_period IS NULL AND listing_type = 'rent') AS missing_period,
                   COUNT(*) FILTER (WHERE listed_at IS NULL) AS missing_listed_at,
                   COUNT(*) FILTER (WHERE price_period = 'month' AND price_ngn > 10000000) AS implausible_monthly,
                   COUNT(*) FILTER (WHERE (price_period = 'year' OR price_period IS NULL) AND price_ngn < 300000 AND listing_type = 'rent') AS implausible_yearly,
                   COUNT(*) FILTER (WHERE created_at >= now() - interval '7 days') AS added_7d,
                   MAX(listed_at) AS newest_listing, MIN(listed_at) AS oldest_listing
            FROM listings
        """)).mappings().one()
        by_source = _rows(conn, "SELECT source, COUNT(*) AS c FROM listings WHERE is_duplicate_of IS NULL GROUP BY 1 ORDER BY 2 DESC", {})
        stale_areas = _rows(conn, """
            SELECT area, listing_count, last_ingested_at FROM area_freshness
            WHERE last_ingested_at < now() - interval '14 days' ORDER BY listing_count DESC LIMIT 10
        """, {})
        last_run = conn.execute(text("SELECT * FROM scraper_runs ORDER BY started_at DESC LIMIT 1")).mappings().first()
        runs_30d = conn.execute(text("""
            SELECT COUNT(*) AS runs, COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                   COALESCE(SUM(listings_added), 0) AS added, COALESCE(SUM(listings_updated), 0) AS updated
            FROM scraper_runs WHERE started_at >= now() - interval '30 days'
        """)).mappings().one()
        saved = conn.execute(text("""
            SELECT (SELECT COUNT(*) FROM saved_searches) AS saved_searches,
                   (SELECT COUNT(DISTINCT client_id) FROM saved_searches) AS saved_search_clients,
                   (SELECT COUNT(*) FROM favorites) AS favorites,
                   (SELECT COUNT(DISTINCT client_id) FROM favorites) AS favorite_clients,
                   (SELECT COUNT(*) FROM area_guides) AS guides,
                   (SELECT MAX(week_start) FROM price_snapshots) AS latest_snapshot_week,
                   (SELECT COUNT(DISTINCT week_start) FROM price_snapshots) AS snapshot_weeks
        """)).mappings().one()
        db_size = conn.execute(text("SELECT pg_database_size(current_database())")).scalar()

    now = datetime.now(UTC)
    req = dict(traffic)
    total = req["requests"] or 0
    last = dict(last_run) if last_run else None
    health = []
    if total and req["server_errors"] / total > 0.02:
        health.append({"level": "critical", "text": f"{req['server_errors']} server errors ({req['server_errors'] / total:.1%} of requests)"})
    if any(a["failures"] for a in ai):
        health.append({"level": "warning", "text": f"{sum(a['failures'] for a in ai)} failed AI or search calls"})
    blocked_note = ""
    if last and "BLOCKED:" in (last.get("log_excerpt") or ""):
        blocked_note = " · " + (last["log_excerpt"].split("BLOCKED:", 1)[1].split("\n", 1)[0].strip())
    if last and last["status"] == "failed":
        health.append({"level": "critical", "text": f"Last scrape run failed{blocked_note}"})
    elif last and last["status"] == "partial":
        health.append({"level": "warning", "text": f"Last scrape run was partial{blocked_note}"})
    elif last and last["status"] == "interrupted":
        health.append({"level": "warning", "text": "Last scrape run was stopped before finishing. Pages it saved are kept; the next run continues from there."})
    if last and last["started_at"] and last["started_at"] < now - timedelta(days=8):
        health.append({"level": "warning", "text": "No scrape run in over 8 days"})
    if not last:
        health.append({"level": "warning", "text": "No scrape run recorded yet"})
    if data["implausible_monthly"] or data["implausible_yearly"]:
        health.append({"level": "warning", "text": f"{data['implausible_monthly'] + data['implausible_yearly']} listings have a price period that looks wrong; they skew area ranges"})
    if data["missing_period"]:
        health.append({"level": "info", "text": f"{data['missing_period']} rent listings have no price period (treated as yearly)"})
    return {
        "window_hours": hours,
        "health": health,
        "traffic": {**req, "p50_ms": _f(req["p50_ms"]), "p95_ms": _f(req["p95_ms"]),
                    "error_rate": round(req["server_errors"] / total * 100, 2) if total else 0.0},
        "routes": [{**r, "p50_ms": _f(r["p50_ms"]), "p95_ms": _f(r["p95_ms"])} for r in routes],
        "devices": devices,
        "ai": [{**a, "avg_ms": _f(a["avg_ms"]), "p95_ms": _f(a["p95_ms"])} for a in ai],
        "errors_by_source": errors_by_source,
        "questions": {**dict(questions), "avg_ms": _f(questions["avg_ms"]), "p95_ms": _f(questions["p95_ms"])},
        "unanswered_areas": unanswered_areas,
        "stages": [{**s, "avg_ms": _f(s["avg_ms"]), "p95_ms": _f(s["p95_ms"])} for s in stages],
        "features": features,
        "data": {**dict(data), "by_source": by_source, "stale_areas": stale_areas, "db_size_bytes": db_size},
        "scraper": {"last_run": last, **dict(runs_30d), "next_scheduled_run": _next_weekly_run(now).isoformat()},
        "engagement": dict(saved),
    }


@router.get("/timeseries")
def timeseries(request: Request, hours: int = Query(168, ge=1, le=24 * 90), admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    unit = "hour" if hours <= 72 else "day"
    p = {"since": _since(hours)}
    with get_engine().connect() as conn:
        reqs = _rows(conn, f"""
            SELECT date_trunc('{unit}', created_at) AS t, COUNT(*) AS requests,
                   COUNT(*) FILTER (WHERE status_code >= 500) AS errors,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
            FROM api_requests WHERE created_at >= :since AND route NOT LIKE '/api/admin%' GROUP BY 1 ORDER BY 1
        """, p)
        qs = _rows(conn, f"SELECT date_trunc('{unit}', created_at) AS t, COUNT(*) AS questions FROM question_events WHERE created_at >= :since GROUP BY 1 ORDER BY 1", p)
    by_t: dict[str, dict] = {}
    for r in reqs:
        by_t[r["t"].isoformat()] = {"t": r["t"].isoformat(), "requests": r["requests"], "errors": r["errors"], "p95_ms": _f(r["p95_ms"]), "questions": 0}
    for q in qs:
        k = q["t"].isoformat()
        by_t.setdefault(k, {"t": k, "requests": 0, "errors": 0, "p95_ms": None, "questions": 0})["questions"] = q["questions"]
    return {"unit": unit, "points": sorted(by_t.values(), key=lambda x: x["t"])}


@router.get("/errors")
def errors(request: Request, source: str | None = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
           admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    where = "WHERE source = :source" if source else ""
    p: dict[str, object] = {"lim": page_size, "off": (page - 1) * page_size, "source": source}
    with get_engine().connect() as conn:
        items = _rows(conn, f"SELECT id, request_id, source, route, error_type, message, stack, created_at FROM error_events {where} ORDER BY created_at DESC LIMIT :lim OFFSET :off", p)
        total = conn.execute(text(f"SELECT COUNT(*) FROM error_events {where}"), p).scalar() or 0
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/ai-calls")
def ai_calls(request: Request, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    p = {"lim": page_size, "off": (page - 1) * page_size}
    with get_engine().connect() as conn:
        items = _rows(conn, "SELECT id, request_id, provider, purpose, model, latency_ms, prompt_tokens, completion_tokens, success, error, created_at FROM llm_calls ORDER BY created_at DESC LIMIT :lim OFFSET :off", p)
        total = conn.execute(text("SELECT COUNT(*) FROM llm_calls")).scalar() or 0
    return {"items": [{**i, "latency_ms": _f(i["latency_ms"])} for i in items], "total": total, "page": page, "page_size": page_size}


@router.get("/requests/{request_id}")
def request_detail(request_id: str, request: Request, admin_session: str | None = Cookie(default=None)):
    """Everything recorded for one request id: HTTP row, question, pipeline stages, AI calls, errors."""
    get_current_admin(request, admin_session)
    p = {"rid": request_id}
    with get_engine().connect() as conn:
        out = {
            "request": _rows(conn, "SELECT * FROM api_requests WHERE request_id = :rid", p),
            "question": _rows(conn, "SELECT * FROM question_events WHERE request_id = :rid", p),
            "stages": _rows(conn, "SELECT stage, query_type, duration_ms, call_counts, success, error, created_at FROM traces WHERE trace_id = :rid ORDER BY created_at", p),
            "ai_calls": _rows(conn, "SELECT provider, purpose, model, latency_ms, prompt_tokens, completion_tokens, success, error, created_at FROM llm_calls WHERE request_id = :rid ORDER BY created_at", p),
            "errors": _rows(conn, "SELECT source, error_type, message, stack, created_at FROM error_events WHERE request_id = :rid ORDER BY created_at", p),
        }
    if not any(out.values()):
        raise HTTPException(status_code=404, detail="Nothing recorded for that request id")
    return out


@router.get("/guides")
def guides(request: Request, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    with get_engine().connect() as conn:
        items = _rows(conn, "SELECT area, model, generated_at, jsonb_array_length(sources) AS source_count FROM area_guides ORDER BY generated_at DESC", {})
    return {"items": items}


@router.post("/guides/{area}/refresh")
def refresh_guide(area: str, request: Request, admin_session: str | None = Cookie(default=None)):
    sess = get_current_admin(request, admin_session)
    require_csrf(request, sess)
    from advice.area_guide import GuideUnavailable, generate_guide

    try:
        g = generate_guide(area)
    except GuideUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "area": area, "sources": len(g["sources"])}
