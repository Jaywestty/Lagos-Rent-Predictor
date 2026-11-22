from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from scraping import scraper
from scraping.runner import ProgressTracker, StopFlag
from tests.test_scraper_recency import _TwoPageFetcher


def test_progress_tracker_records_step_page_and_listing():
    writes: list[dict] = []
    saved = {"added": 3, "updated": 1, "pages": 1}
    tracker = ProgressTracker(7, saved, min_interval=60, writer=writes.append)
    stats = {"kept": 4, "too_old": 2}
    tracker({"event": "step", "step": 2, "total_steps": 6, "site": "npc", "category": "houses", "stats": stats})
    tracker({"event": "page", "page": 1, "page_number": 1, "max_pages": 30, "page_total": 25, "stats": stats})
    tracker({"event": "listing", "page_done": 5, "page_total": 25, "stats": stats})  # throttled, not written
    assert len(writes) == 2
    last = writes[-1]
    assert (last["step"], last["total_steps"], last["site"], last["page_total"]) == (2, 6, "npc", 25)
    assert last["counts"]["kept"] == 4 and last["counts"]["too_old"] == 2
    assert last["saved"] == saved and "heartbeat_at" in last
    tracker.write(force=True)
    assert writes[-1]["page_done"] == 5


def test_progress_writer_failure_never_raises():
    def broken(_):
        raise RuntimeError("db down")

    ProgressTracker(1, {}, writer=broken).phase("scraping")


def test_stop_flag_checks_database_at_most_every_interval():
    calls = []
    flag = StopFlag(1, min_interval=3600, reader=lambda: calls.append(1) or False)
    assert flag() is False and flag() is False
    assert len(calls) == 1
    sticky = StopFlag(1, min_interval=0, reader=lambda: True)
    assert sticky() is True and sticky() is True


def test_scraper_stops_cleanly_and_keeps_finished_pages(tmp_path):
    fake = _TwoPageFetcher(fresh=datetime.now(UTC) - timedelta(days=1))
    saved_pages: list[list[dict]] = []
    progress: list[dict] = []
    stop_after_first_page = {"stop": False}

    def on_page(records):
        saved_pages.append(records)
        stop_after_first_page["stop"] = True

    try:
        with patch.object(scraper, "Fetcher", return_value=fake):
            scraper.crawl("propertypro", "flat-apartment", "rent", max_pages=5, since_days=7,
                          output_csv=tmp_path / "run.csv", on_page=on_page, on_progress=progress.append,
                          should_stop=lambda: stop_after_first_page["stop"])
        raise AssertionError("expected ScrapeStopped")
    except scraper.ScrapeStopped:
        pass
    assert len(saved_pages) == 1 and len(saved_pages[0]) == 2  # page one was saved before stopping
    assert progress[0]["event"] == "page"
    assert {e["event"] for e in progress} >= {"page", "listing"}


def test_admin_scrape_control_requires_auth():
    from fastapi.testclient import TestClient

    from entrypoints.api import app

    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/api/admin/operations/scrape/active").status_code in (401, 403)
    assert c.post("/api/admin/operations/scrape/5/stop").status_code in (401, 403)


def test_run_without_heartbeat_is_alive_only_shortly_after_start():
    from admin import api

    api._run_state.pop("41", None)
    old = {"triggered_by": "admin:me@example.com", "status": "running", "progress": None,
           "started_at": datetime.now(UTC) - timedelta(minutes=30)}
    new = {**old, "started_at": datetime.now(UTC) - timedelta(seconds=20)}
    assert api._run_is_alive(41, old) is False  # e.g. left over from before a restart
    assert api._run_is_alive(41, new) is True  # another server may have just started it


def test_scheduled_run_alive_only_with_recent_heartbeat():
    from admin import api

    fresh = {"triggered_by": "laptop-schedule", "status": "running",
             "progress": {"heartbeat_at": datetime.now(UTC).isoformat()}}
    stale = {"triggered_by": "laptop-schedule", "status": "running",
             "progress": {"heartbeat_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat()}}
    silent = {"triggered_by": "laptop-schedule", "status": "running", "progress": None,
              "started_at": datetime.now(UTC) - timedelta(minutes=10)}
    assert api._run_is_alive(1, fresh) is True
    assert api._run_is_alive(1, stale) is False
    assert api._run_is_alive(1, silent) is False
