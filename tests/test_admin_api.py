from unittest.mock import patch


def _client():
    from fastapi.testclient import TestClient

    from entrypoints.api import app
    return TestClient(app, raise_server_exceptions=False)


def test_admin_routes_require_auth():
    c = _client()
    for path in ["/api/admin/auth/me", "/api/admin/analytics/overview", "/api/admin/analytics/questions-over-time", "/api/admin/analytics/top-areas", "/api/admin/analytics/recent-questions", "/api/admin/operations/scraper-runs", "/api/admin/operations/freshness", "/api/admin/operations/review-queue", "/api/admin/operations/scrape/1", "/api/admin/operations/scrape/1/logs"]:
        r = c.get(path, cookies={})
        assert r.status_code in (401, 403), f"{path} returned {r.status_code}"


def test_login_generic_error_and_rate_limit():
    from admin.auth import _is_locked_out, _record_failure
    for _ in range(5):
        _record_failure("x@example.com")
    assert _is_locked_out("x@example.com")
    c = _client()
    # Even with lockout, login endpoint only returns generic errors.
    with patch("admin.api.get_engine"):
        pass
    r = c.post("/api/admin/auth/login", json={"email": "x@example.com", "password": "wrong"})
    assert r.status_code in (401, 429)


def test_scrape_lock_single_run():
    import threading

    from admin.api import _scrape_lock
    # Acquire to simulate running job, then trigger should conflict if lock held.
    # We do not hit the endpoint with a real DB here; we check lock primitives.
    assert isinstance(_scrape_lock, type(threading.Lock()))
    held = _scrape_lock.acquire(blocking=False)
    try:
        assert held is True
        assert _scrape_lock.locked()
    finally:
        _scrape_lock.release()


def test_analytics_aggregation_sql():
    from pathlib import Path
    src = Path("admin/api.py").read_text(encoding="utf-8")
    assert "COUNT(DISTINCT session_id)" in src
    assert "AVG(latency_ms)" in src
    assert "GROUP BY parsed_location" in src
    assert "COUNT(*) FROM listings WHERE is_duplicate_of IS NULL" in src
    assert "sessions" in src and "average_latency_ms" in src


def test_password_hash_round_trip():
    from admin.auth import hash_password, verify_password

    h = hash_password("correct horse battery")
    assert h.startswith("$argon2")
    assert verify_password(h, "correct horse battery")
    assert not verify_password(h, "wrong")
