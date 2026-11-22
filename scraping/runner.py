"""Callable scrape-and-load job used by the weekly scheduler, the CLI and the admin API.

One call = one scraper_runs row, updated in place as the job moves through
running -> success / partial / failed (or interrupted, if the process is killed).

Saving is incremental: every scraped page is written to the database straight
away, and the run row shows live progress. Stopping a run therefore never loses
finished pages. Starting again skips listings already saved in the last
`resume_hours`, so a restarted run continues from where the stopped one ended
instead of re-fetching everything. A final pass over the run's CSV applies
cross-page de-duplication and writes the quality report.

A site that refuses access (HTTP 403, for example when it blocks cloud servers)
is skipped for the run, never worked around. If one site is blocked but another
delivered, the run is "partial". If nothing was collected because of blocks,
the run is "failed" and run_scrape raises, so a scheduled job shows as failed.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from analytics_logging import record_scraper_run, update_scraper_run
from db import get_engine

RUNS_DIR = Path("data/raw/runs")
DEFAULT_SINCE_DAYS = 7
DEFAULT_MAX_PAGES = 30
# Listings saved this recently are not fetched again. Shorter than a week, so each
# weekly run still refreshes last week's prices, but long enough to cover a restart.
DEFAULT_RESUME_HOURS = 72
# The scheduled task is stopped by Windows after 6 hours, so an older "running" row is dead.
STALE_RUN_HOURS = 7


class ScrapeBlocked(RuntimeError):
    pass


class ProgressTracker:
    """Turns scraper progress events into one JSON record on the run row, for the admin progress bar.

    Step and page changes are written at once; per-listing updates at most every
    `min_interval` seconds so the database is not hammered.
    """

    COUNT_KEYS = ("discovered", "kept", "already_have", "too_old", "no_date", "no_price", "fetch_failed", "pages_failed")

    def __init__(self, run_id: int, saved: dict, min_interval: float = 3.0, writer=None):
        self.run_id = run_id
        self.saved = saved
        self.min_interval = min_interval
        self.writer = writer or (lambda progress: update_scraper_run(run_id, progress=progress))
        self._last_write = 0.0
        self.state: dict = {
            "phase": "scraping", "step": 0, "total_steps": 0, "site": None, "category": None,
            "page": None, "page_number": 0, "max_pages": None, "page_done": 0, "page_total": 0,
            "started_at": datetime.now(UTC).isoformat(),
        }

    def __call__(self, event: dict) -> None:
        kind = event.get("event")
        stats = event.get("stats") or {}
        if kind == "step":
            self.state.update(step=event["step"], total_steps=event["total_steps"], site=event["site"],
                              category=event["category"], page=None, page_number=0, page_done=0, page_total=0)
        elif kind == "page":
            self.state.update(page=event["page"], page_number=event["page_number"], max_pages=event["max_pages"],
                              page_done=0, page_total=event["page_total"])
        elif kind == "listing":
            self.state.update(page_done=event["page_done"], page_total=event["page_total"])
        self.state["counts"] = {k: stats.get(k, 0) for k in self.COUNT_KEYS}
        self.write(force=kind in ("step", "page"))

    def phase(self, name: str) -> None:
        self.state["phase"] = name
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < self.min_interval:
            return
        self._last_write = now
        self.state["saved"] = dict(self.saved)
        self.state["heartbeat_at"] = datetime.now(UTC).isoformat()
        try:
            self.writer(dict(self.state))
        except Exception as exc:  # progress display must never stop a scrape
            logger.warning("progress update failed: {}", exc)


def run_status(stats: dict) -> str:
    """success: no site refused us. partial: a site was blocked but another delivered. failed: nothing collected."""
    if not stats.get("blocked_sites"):
        return "success"
    return "partial" if stats.get("kept", 0) + stats.get("already_have", 0) > 0 else "failed"


class StopFlag:
    """Asks the database whether an admin pressed Stop for this run, at most every `min_interval` seconds."""

    def __init__(self, run_id: int, min_interval: float = 4.0, reader=None):
        self.run_id = run_id
        self.min_interval = min_interval
        self.reader = reader or self._read
        self._last_check = 0.0
        self._stop = False

    def _read(self) -> bool:
        with get_engine().connect() as conn:
            return bool(conn.execute(text("SELECT stop_requested_at IS NOT NULL FROM scraper_runs WHERE id = :id"),
                                     {"id": self.run_id}).scalar())

    def __call__(self) -> bool:
        if self._stop:
            return True
        now = time.monotonic()
        if now - self._last_check >= self.min_interval:
            self._last_check = now
            self._stop = self.reader()
        return self._stop


def request_stop(run_id: int, engine=None) -> bool:
    """Ask a running scrape to stop after the listing it is on. True if the run was running."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        n = conn.execute(text("UPDATE scraper_runs SET stop_requested_at = COALESCE(stop_requested_at, now()) "
                              "WHERE id = :id AND status IN ('running', 'queued')"), {"id": run_id}).rowcount
    return bool(n)


def mark_interrupted_runs(engine=None, older_than_hours: int = STALE_RUN_HOURS) -> list[int]:
    """Rows still 'running' or 'queued' long after any run could last were killed mid-way."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("""
            UPDATE scraper_runs
            SET status = 'interrupted', completed_at = now(),
                log_excerpt = COALESCE(log_excerpt, '') || CASE WHEN COALESCE(log_excerpt, '') = '' THEN '' ELSE E'\\n' END
                              || 'Stopped before finishing. Pages saved before the stop are kept.'
            WHERE status IN ('running', 'queued') AND started_at < now() - make_interval(hours => :h)
            RETURNING id
        """), {"h": older_than_hours}).scalars().all()
    if rows:
        logger.info("Marked runs as interrupted: {}", rows)
    return list(rows)


def mark_run_interrupted(run_id: int, loaded: dict | None = None, engine=None) -> bool:
    """Close one stopped run whose CSV was loaded by hand (entrypoints.cli load-run)."""
    eng = engine or get_engine()
    note = "Stopped before finishing; its CSV was loaded afterwards with load-run."
    if loaded:
        note += f" Loaded: {loaded.get('added', 0)} new, {loaded.get('updated', 0)} updated."
    with eng.begin() as conn:
        n = conn.execute(text("""
            UPDATE scraper_runs SET status = 'interrupted', completed_at = COALESCE(completed_at, now()),
                   listings_added = GREATEST(listings_added, :added), log_excerpt = :note
            WHERE id = :id AND status IN ('running', 'queued')
        """), {"id": run_id, "added": (loaded or {}).get("added", 0), "note": note}).rowcount
    return bool(n)


def recently_collected(hours: int, engine=None) -> dict[str, datetime | None]:
    """URL -> listing date for listings saved in the last `hours`; those are skipped on resume."""
    if hours <= 0:
        return {}
    eng = engine or get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT url, listed_at FROM listings
            WHERE listing_type = 'rent' AND scraped_at >= now() - make_interval(hours => :h)
        """), {"h": hours}).all()
    return {r[0]: r[1] for r in rows}


def run_scrape(triggered_by: str = "schedule", max_pages: int = DEFAULT_MAX_PAGES, since_days: int = DEFAULT_SINCE_DAYS,
               run_id: int | None = None, site: str = "all", category: str = "all",
               resume_hours: int = DEFAULT_RESUME_HOURS) -> dict:
    """Scrape recent rent listings, saving each page as it goes. Returns a summary including run_id and status."""
    mark_interrupted_runs()
    if run_id is None:
        run_id = record_scraper_run(triggered_by=triggered_by, status="running", completed=False)
    update_scraper_run(run_id, status="running", since_days=since_days)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RUNS_DIR / f"run_{run_id}_{stamp}.csv"
    saved = {"added": 0, "updated": 0, "pages": 0}
    progress = ProgressTracker(run_id, saved)
    progress.phase("starting")

    try:
        from pipeline.load_data import load_csv_to_db, load_records
        from query.market import write_snapshots
        from scraping.scraper import ScrapeStopped
        from scraping.scraper import main as scraper_main

        known = recently_collected(resume_hours)
        if known:
            logger.info("Resuming: {} listings saved in the last {}h will not be fetched again", len(known), resume_hours)

        def save_page(records: list[dict]) -> None:
            result = load_records(records)
            saved["added"] += result["added"]
            saved["updated"] += result["updated"]
            saved["pages"] += 1
            update_scraper_run(run_id, listings_added=saved["added"], listings_updated=saved["updated"],
                               log_excerpt=f"In progress: {saved['pages']} pages saved, "
                                           f"{saved['added']} new and {saved['updated']} updated listings so far")
            progress.write(force=True)

        progress.phase("scraping")

        stopped = False
        try:
            stats = scraper_main([
                "--site", site, "--category", category, "--type", "rent",
                "--max-pages", str(max_pages), "--since-days", str(since_days), "--output", str(csv_path),
            ], on_page=save_page, known=known, on_progress=progress, should_stop=StopFlag(run_id))
        except ScrapeStopped:
            # Stopped from admin. Everything finished so far is saved; tidy up and record it.
            stopped = True
            logger.info("scrape run {} stopped on request", run_id)
            stats = {k: 0 for k in ("discovered", "kept", "already_have", "too_old", "no_date", "no_price",
                                    "fetch_failed", "pages_failed", "page_load_errors")}
            stats.update(progress.state.get("counts") or {})
            stats["blocked_sites"] = []

        progress.phase("finishing")
        # Final pass: cross-page de-duplication and the quality report over everything this run fetched.
        final = load_csv_to_db(str(csv_path), report_out=str(RUNS_DIR / f"run_{run_id}_quality.json"))
        snapshots = write_snapshots()
        blocked = stats.get("blocked_sites") or []
        status = "stopped" if stopped else run_status(stats)
        lines = [f"window={since_days}d max_pages={max_pages} site={site} category={category} resume={resume_hours}h"]
        if blocked:
            lines.append(f"BLOCKED: {', '.join(blocked)} refused access (HTTP 403) from this machine")
        lines += [
            f"saved while running: {saved['added']} new, {saved['updated']} updated, {saved['pages']} pages",
            f"skipped (already saved recently): {stats.get('already_have', 0)}",
            f"scrape={stats}", f"final pass={final}", f"snapshots={snapshots}", f"csv={csv_path}",
        ]
        excerpt = "\n".join(lines)
        update_scraper_run(
            run_id,
            completed=True,
            status=status,
            listings_discovered=stats["discovered"],
            listings_added=saved["added"],
            listings_updated=saved["updated"],
            duplicates_skipped=final["duplicates"],
            validation_failures=stats["no_price"] + stats["no_date"],
            listings_too_old=stats["too_old"],
            error_count=stats["fetch_failed"] + stats.get("pages_failed", 0) + stats.get("page_load_errors", 0),
            log_excerpt=excerpt,
        )
        progress.phase("stopped" if stopped else "done")
        logger.info("scrape run {} {}: {}", run_id, status, " | ".join(lines))
        if blocked:
            from observability.events import drain, record_error

            record_error("scraper", f"Sites refused access this run: {', '.join(blocked)}")
            drain()
    except Exception as exc:
        logger.exception("scrape run {} failed", run_id)
        from observability.events import drain, record_error

        record_error("scraper", exc)
        drain()
        progress.phase("failed")
        update_scraper_run(run_id, completed=True, status="failed", error_count=1,
                           listings_added=saved["added"], listings_updated=saved["updated"],
                           log_excerpt=f"{exc!r}\nSaved before the failure: {saved['added']} new, {saved['updated']} updated")
        raise

    if status == "failed":
        # Exit non-zero so the scheduled job shows as failed.
        raise ScrapeBlocked(f"No listings collected; sites refused access: {', '.join(blocked)}")
    return {"run_id": run_id, "status": status, "saved": saved, "scrape": stats, "load": final, "csv": str(csv_path)}
