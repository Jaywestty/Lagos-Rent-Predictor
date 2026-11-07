from loguru import logger

from analytics_logging import record_scraper_run


def run_scrape(triggered_by="schedule", max_pages=5):
    run_id = record_scraper_run(triggered_by=triggered_by, status="running", completed=False)
    try:
        import sys

        from scraping.scraper import main as scraper_main

        old_argv = sys.argv
        sys.argv = ["scraper.py", "--site", "all", "--category", "all", "--type", "rent", "--max-pages", str(max_pages)]
        try:
            scraper_main()
        finally:
            sys.argv = old_argv
        record_scraper_run(triggered_by=triggered_by, status="success", completed=True)
        return run_id
    except Exception as exc:
        logger.error("scrape failed: {}", exc)
        record_scraper_run(triggered_by=triggered_by, status="failed", completed=True, error_count=1, log_excerpt=str(exc)[:8000])
        raise
