"""Callable scrape-and-load job used by the weekly scheduler, the CLI and the admin API.

One call = one scraper_runs row, updated in place as the job moves through
running -> success / failed. The job scrapes only listings from the last
`since_days` days into a per-run CSV, then loads that CSV into Postgres.
"""

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from analytics_logging import record_scraper_run, update_scraper_run

RUNS_DIR = Path("data/raw/runs")
DEFAULT_SINCE_DAYS = 7
DEFAULT_MAX_PAGES = 30


def run_scrape(triggered_by: str = "schedule", max_pages: int = DEFAULT_MAX_PAGES, since_days: int = DEFAULT_SINCE_DAYS,
               run_id: int | None = None, site: str = "all", category: str = "all") -> dict:
    """Scrape recent rent listings and load them. Returns a summary dict including run_id."""
    if run_id is None:
        run_id = record_scraper_run(triggered_by=triggered_by, status="running", completed=False)
    update_scraper_run(run_id, status="running", since_days=since_days)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RUNS_DIR / f"run_{run_id}_{stamp}.csv"
    try:
        from pipeline.load_data import load_csv_to_db
        from scraping.scraper import main as scraper_main

        stats = scraper_main([
            "--site", site, "--category", category, "--type", "rent",
            "--max-pages", str(max_pages), "--since-days", str(since_days), "--output", str(csv_path),
        ])
        loaded = load_csv_to_db(str(csv_path), report_out=str(RUNS_DIR / f"run_{run_id}_quality.json"))
        excerpt = (
            f"window={since_days}d max_pages={max_pages} site={site} category={category}\n"
            f"scrape={stats}\nload={loaded}\ncsv={csv_path}"
        )
        update_scraper_run(
            run_id,
            completed=True,
            status="success",
            listings_discovered=stats["discovered"],
            listings_added=loaded["added"],
            listings_updated=loaded["updated"],
            duplicates_skipped=loaded["duplicates"],
            validation_failures=stats["no_price"] + stats["no_date"],
            listings_too_old=stats["too_old"],
            error_count=stats["fetch_failed"],
            log_excerpt=excerpt,
        )
        logger.info("scrape run {} done: {}", run_id, excerpt.replace("\n", " | "))
        return {"run_id": run_id, "scrape": stats, "load": loaded, "csv": str(csv_path)}
    except Exception as exc:
        logger.exception("scrape run {} failed", run_id)
        update_scraper_run(run_id, completed=True, status="failed", error_count=1, log_excerpt=repr(exc))
        raise
