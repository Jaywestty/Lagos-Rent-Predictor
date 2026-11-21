from datetime import UTC, date, datetime
from unittest.mock import patch


def _client():
    from fastapi.testclient import TestClient

    from entrypoints.api import app
    return TestClient(app, raise_server_exceptions=False)


def test_week_start_is_monday():
    from query.market import _week_start

    assert _week_start(date(2026, 11, 21)) == date(2026, 11, 16)
    assert _week_start(datetime(2026, 11, 16, 9, tzinfo=UTC)) == date(2026, 11, 16)


def test_next_weekly_run_matches_workflow_cron():
    from admin.observability import _next_weekly_run

    assert _next_weekly_run(datetime(2026, 11, 21, 12, tzinfo=UTC)) == datetime(2026, 11, 23, 1, tzinfo=UTC)
    assert _next_weekly_run(datetime(2026, 11, 23, 0, 30, tzinfo=UTC)) == datetime(2026, 11, 23, 1, tzinfo=UTC)
    assert _next_weekly_run(datetime(2026, 11, 23, 1, 30, tzinfo=UTC)) == datetime(2026, 11, 30, 1, tzinfo=UTC)


def test_compare_uses_bucket_for_beds():
    fake = {"area": "Yaba", "listing_count": 3, "status": "low_data", "confidence": "insufficient",
            "overall": {"median": 1}, "breakdown": {"two_bedroom": {"median": 2}}, "data_as_of": None}
    with patch("query.market.area_stats", return_value=fake), patch("query.market.trends", return_value={"points": []}):
        from query.market import compare

        out = compare(["Yaba", "Lekki"], beds=2)
    assert out["bucket"] == "two_bedroom"
    assert [a["focus"] for a in out["areas"]] == [{"median": 2}, {"median": 2}]


def test_saved_items_require_a_client_id():
    c = _client()
    assert c.get("/api/me/saved-searches").status_code == 400
    assert c.get("/api/me/favorites", headers={"X-Client-Id": "not-a-uuid"}).status_code == 400


def test_favorites_only_accept_supported_sites():
    c = _client()
    r = c.post("/api/me/favorites", headers={"X-Client-Id": "6f1c2d3e-0000-4000-8000-000000000001"},
               json={"url": "https://evil.example.com/listing"})
    assert r.status_code == 400


def test_unknown_usage_events_are_rejected():
    assert _client().post("/api/events", json={"feature": "anything"}).status_code == 400


def test_admin_observability_requires_auth():
    c = _client()
    for path in ["/api/admin/observability/overview", "/api/admin/observability/timeseries",
                 "/api/admin/observability/errors", "/api/admin/observability/ai-calls", "/api/admin/observability/guides"]:
        assert c.get(path).status_code in (401, 403), path


def test_guide_sections_are_cleaned_and_source_ids_bounded():
    from advice.area_guide import _clean

    raw = {"overview": {"text": " Busy tech hub. ", "sources": [1, 9, "2"]}, "power": {"text": None, "sources": []},
           "watch_outs": ["Flooding on main roads.", 3, ""], "watch_out_sources": [2]}
    out = _clean(raw, n_sources=3)
    overview = next(s for s in out["sections"] if s["key"] == "overview")
    assert overview == {"key": "overview", "title": "Overview", "text": "Busy tech hub.", "sources": [1, 2]}
    assert next(s for s in out["sections"] if s["key"] == "power")["text"] is None
    assert out["watch_outs"] == ["Flooding on main roads."]
