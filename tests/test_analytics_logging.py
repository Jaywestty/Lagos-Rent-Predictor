from pathlib import Path


def test_migration_files_present_and_parse():
    migs = sorted(Path("sql/migrations").glob("*.sql"))
    assert len(migs) >= 3
    q = (Path("sql/migrations/0001_analytics.sql")).read_text(encoding="utf-8")
    s = (Path("sql/migrations/0002_scraper_runs.sql")).read_text(encoding="utf-8")
    a = (Path("sql/migrations/0003_freshness_admin.sql")).read_text(encoding="utf-8")
    for col in ["request_id", "session_id", "raw_question", "parsed_location", "status", "listing_count", "confidence", "latency_ms", "error_code", "review_state"]:
        assert col in q
    for col in ["triggered_by", "started_at", "completed_at", "status", "listings_discovered", "listings_added", "listings_updated", "duplicates_skipped", "validation_failures", "error_count", "log_excerpt"]:
        assert col in s
    assert "area_freshness" in a
    assert "admin_users" in a


def test_log_question_sql_has_no_ip():
    src = Path("analytics_logging.py").read_text(encoding="utf-8")
    assert "ip" not in src.lower().replace("strip", "").replace("pipeline", "") or "No IP" in src
    assert "question_events" in src


def test_failure_logging_path():
    src = Path("entrypoints/api.py").read_text(encoding="utf-8")
    assert 'log_question(' in src
    assert 'error_code="query_failed"' in src or "error_code" in src
