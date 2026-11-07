import threading
import time
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Cookie, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text

from admin.auth import SESSION_COOKIE, create_admin, get_current_admin, login, logout, require_csrf
from db import get_engine

router = APIRouter(prefix="/admin", tags=["admin"])

_scrape_lock = threading.Lock()
_last_scrape_at: float = 0.0
_run_state: dict[str, dict] = {}
_jobs: dict[int, dict] = {}


class LoginIn(BaseModel):
    email: str
    password: str


class ReviewIn(BaseModel):
    review_state: str = "reviewed"
    review_note: str | None = None
    flag_for_coverage: bool = False


def _paginate(rows, total, page, page_size):
    return {"items": rows, "total": total, "page": page, "page_size": page_size}


def _filter_sql(params: dict, base_where: str) -> tuple[dict[str, object], str]:
    sql_params: dict[str, object] = {}
    clauses = [base_where]
    if params.get("from"):
        clauses.append("created_at >= :from_ts")
        sql_params["from_ts"] = params["from"]
    if params.get("to"):
        clauses.append("created_at <= :to_ts")
        sql_params["to_ts"] = params["to"]
    if params.get("area"):
        clauses.append("area = :area")
        sql_params["area"] = params["area"]
    if params.get("property_type"):
        clauses.append("property_type = :pt")
        sql_params["pt"] = params["property_type"]
    if params.get("status"):
        clauses.append("status = :status")
        sql_params["status"] = params["status"]
    return sql_params, " AND ".join(c for c in clauses if c)


@router.post("/auth/login")
def auth_login(body: LoginIn, response: Response):
    sid, csrf = login(body.email, body.password)
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, secure=True, samesite="lax", path="/admin", max_age=12 * 3600)
    return {"ok": True, "csrf_token": csrf}


@router.post("/auth/logout")
def auth_logout(request: Request, admin_session: str | None = Cookie(default=None)):
    sess = get_current_admin(request, admin_session)
    require_csrf(request, sess)
    logout(admin_session)
    return {"ok": True}


@router.get("/auth/me")
def auth_me(request: Request, admin_session: str | None = Cookie(default=None)):
    sess = get_current_admin(request, admin_session)
    return {"email": sess.get("email"), "role": sess.get("role"), "admin_id": sess.get("admin_id")}


@router.get("/analytics/overview")
def analytics_overview(request: Request, from_date: str | None = None, to_date: str | None = None, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        total_q = conn.execute(text("SELECT COUNT(*) FROM question_events")).scalar() or 0
        sessions = conn.execute(text("SELECT COUNT(DISTINCT session_id) FROM question_events WHERE session_id IS NOT NULL")).scalar() or 0
        top = conn.execute(text("SELECT parsed_location, COUNT(*) AS c FROM question_events WHERE parsed_location IS NOT NULL GROUP BY parsed_location ORDER BY c DESC LIMIT 1")).mappings().first()
        low = conn.execute(text("SELECT COUNT(*) FROM question_events WHERE status IN ('no_data','low_data')")).scalar() or 0
        no_data = conn.execute(text("SELECT COUNT(*) FROM question_events WHERE status = 'no_data'")).scalar() or 0
        low_only = conn.execute(text("SELECT COUNT(*) FROM question_events WHERE status = 'low_data'")).scalar() or 0
        avg_lat = conn.execute(text("SELECT COALESCE(AVG(latency_ms), 0) FROM question_events WHERE status NOT IN ('error')")).scalar() or 0
        listings = conn.execute(text("SELECT COUNT(*) FROM listings WHERE is_duplicate_of IS NULL")).scalar() or 0
        last_scrape = conn.execute(text("SELECT MAX(completed_at) FROM scraper_runs")).scalar()
    return {
        "total_questions": total_q,
        "sessions": sessions,
        "top_area": {"area": top["parsed_location"], "count": top["c"]} if top else None,
        "no_or_low_data_count": low,
        "no_data_count": no_data,
        "low_data_count": low_only,
        "no_or_low_data_percentage": round((low / total_q * 100), 2) if total_q else 0.0,
        "average_latency_ms": float(avg_lat),
        "valid_listing_count": listings,
        "last_scrape_at": str(last_scrape) if last_scrape else None,
    }


@router.get("/analytics/questions-over-time")
def questions_over_time(request: Request, page: int = 1, page_size: int = 20, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DATE(created_at) AS day, COUNT(*) AS c FROM question_events GROUP BY DATE(created_at) ORDER BY day DESC LIMIT :lim OFFSET :off"),
            {"lim": page_size, "off": (page - 1) * page_size},
        ).mappings().all()
        total = conn.execute(text("SELECT COUNT(DISTINCT DATE(created_at)) FROM question_events")).scalar() or 0
    return _paginate([dict(r) for r in rows], total, page, page_size)


@router.get("/analytics/top-areas")
def top_areas(request: Request, page: int = 1, page_size: int = 10, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT parsed_location AS area, COUNT(*) AS c FROM question_events WHERE parsed_location IS NOT NULL GROUP BY parsed_location ORDER BY c DESC LIMIT :lim OFFSET :off"),
            {"lim": page_size, "off": (page - 1) * page_size},
        ).mappings().all()
        total = conn.execute(text("SELECT COUNT(DISTINCT parsed_location) FROM question_events WHERE parsed_location IS NOT NULL")).scalar() or 0
    return _paginate([dict(r) for r in rows], total, page, page_size)


@router.get("/analytics/recent-questions")
def recent_questions(request: Request, status: str | None = None, page: int = 1, page_size: int = 20, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    where = "1=1" + (" AND status = :status" if status else "")
    params: dict[str, object] = {"lim": page_size, "off": (page - 1) * page_size}
    if status:
        params["status"] = status
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT id, request_id, session_id, created_at, raw_question, parsed_location, status, listing_count, confidence FROM question_events WHERE {where} ORDER BY created_at DESC LIMIT :lim OFFSET :off"),
            params,
        ).mappings().all()
        total = conn.execute(text(f"SELECT COUNT(*) FROM question_events WHERE {where}"), params).scalar() or 0
    return _paginate([dict(r) for r in rows], total, page, page_size)


@router.get("/operations/scraper-runs")
def scraper_runs(request: Request, page: int = 1, page_size: int = 20, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, triggered_by, started_at, completed_at, status, listings_discovered, listings_added, listings_updated, duplicates_skipped, validation_failures, error_count, log_excerpt FROM scraper_runs ORDER BY started_at DESC LIMIT :lim OFFSET :off"),
            {"lim": page_size, "off": (page - 1) * page_size},
        ).mappings().all()
        total = conn.execute(text("SELECT COUNT(*) FROM scraper_runs")).scalar() or 0
    return _paginate([dict(r) for r in rows], total, page, page_size)


@router.get("/operations/freshness")
def freshness(request: Request, page: int = 1, page_size: int = 50, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT area, listing_count, last_ingested_at FROM area_freshness ORDER BY listing_count DESC LIMIT :lim OFFSET :off"), {"lim": page_size, "off": (page - 1) * page_size}).mappings().all()
        total = conn.execute(text("SELECT COUNT(*) FROM area_freshness")).scalar() or 0
    out = []
    for r in rows:
        last = r["last_ingested_at"]
        if last is None:
            bucket = "Missing"
        else:
            try:
                dt = last if isinstance(last, datetime) else datetime.fromisoformat(str(last))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                age_h = (datetime.now(UTC) - dt).total_seconds() / 3600
                bucket = "Fresh" if age_h < 24 else ("Aging" if age_h <= 72 else "Stale")
            except Exception:
                bucket = "Missing"
        out.append({**dict(r), "freshness": bucket})
    return _paginate(out, total, page, page_size)


@router.get("/operations/review-queue")
def review_queue(request: Request, status: str | None = None, page: int = 1, page_size: int = 20, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    where = "review_state = 'pending'" + (" AND status = :status" if status else "")
    params: dict[str, object] = {"lim": page_size, "off": (page - 1) * page_size}
    if status:
        params["status"] = status
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT id, request_id, raw_question, parsed_location, status, created_at, review_state FROM question_events WHERE {where} ORDER BY created_at DESC LIMIT :lim OFFSET :off"),
            params,
        ).mappings().all()
        total = conn.execute(text(f"SELECT COUNT(*) FROM question_events WHERE {where}"), params).scalar() or 0
    return _paginate([dict(r) for r in rows], total, page, page_size)


def _run_scrape_job(job_id: int, triggered_by: str) -> None:
    try:
        from scraping.runner import run_scrape
        run_scrape(triggered_by=triggered_by, max_pages=5)
    except Exception as exc:
        logger.error("admin scrape failed: {}", exc)
        _run_state[str(job_id)] = {"state": "failed", "error": str(exc)[:500]}
    else:
        _run_state[str(job_id)]["state"] = "success"
    finally:
        _scrape_lock.release()


@router.post("/operations/scrape")
def trigger_scrape(request: Request, background: BackgroundTasks, admin_session: str | None = Cookie(default=None)):
    global _last_scrape_at
    sess = get_current_admin(request, admin_session)
    require_csrf(request, sess)
    now = time.monotonic()
    if not _scrape_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Scrape already running")
    try:
        if now - _last_scrape_at < 60:
            _scrape_lock.release()
            raise HTTPException(status_code=429, detail="Cooldown: scrape already triggered recently")
        from analytics_logging import record_scraper_run
        job_id = record_scraper_run(triggered_by=f"admin:{sess.get('email')}", status="queued", completed=False)
        _last_scrape_at = now
        _run_state[str(job_id)] = {"state": "queued"}
        background.add_task(_run_scrape_job, job_id, f"admin:{sess.get('email')}")
        return {"job_id": job_id, "triggered_by": f"admin:{sess.get('email')}"}
    except HTTPException:
        raise
    except Exception as exc:
        _scrape_lock.release()
        raise HTTPException(status_code=500, detail="Could not queue scrape") from exc


@router.get("/operations/scrape/{job_id}")
def scrape_status(job_id: int, request: Request, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, triggered_by, started_at, completed_at, status, listings_discovered, listings_added, listings_updated, duplicates_skipped, validation_failures, error_count, log_excerpt FROM scraper_runs WHERE id = :id"), {"id": job_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    state = _run_state.get(str(job_id), {}).get("state", dict(row).get("status"))
    return {**dict(row), "job_state": state}


@router.get("/operations/scrape/{job_id}/logs")
def scrape_logs(job_id: int, request: Request, admin_session: str | None = Cookie(default=None)):
    get_current_admin(request, admin_session)
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT log_excerpt FROM scraper_runs WHERE id = :id"), {"id": job_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "log_excerpt": row["log_excerpt"] or ""}


@router.patch("/operations/review/{event_id}")
def review_event(event_id: int, body: ReviewIn, request: Request, admin_session: str | None = Cookie(default=None)):
    sess = get_current_admin(request, admin_session)
    require_csrf(request, sess)
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("UPDATE question_events SET review_state = :state, review_note = :note WHERE id = :id RETURNING id"),
            {"state": body.review_state, "note": body.review_note, "id": event_id},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    if body.flag_for_coverage:
        logger.info("Area flagged for coverage from review item {}", event_id)
    return {"ok": True, "id": event_id, "review_state": body.review_state}


def create_first_admin_from_env() -> int:
    import os
    email = os.getenv("ADMIN_EMAIL") or os.getenv("ADMIN_USER")
    password = os.getenv("ADMIN_PASSWORD")
    if not email or not password:
        raise RuntimeError("ADMIN_EMAIL and ADMIN_PASSWORD must be set")
    return create_admin(email, password)
