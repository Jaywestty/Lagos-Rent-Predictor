# Next milestone: hosted database and CI scraping

Goal: move scraped listings from local Postgres and manual loads to hosted Postgres with a scheduled CI job that runs the scraper and loader on a timetable.

Seams already in place:

- Callable scraper: scraping/runner.py run_scrape(triggered_by, max_pages) writes scraper_runs rows and returns the run id. It wraps scraping/scraper.py main with safe argv handling, so CI can call it directly without shelling.
- Scraper runs table: sql/migrations/0002_scraper_runs.sql defines triggered_by, started_at, completed_at, status (queued, running, success, failed), listings_discovered, listings_added, listings_updated, duplicates_skipped, validation_failures, error_count and log_excerpt.
- Analytics logging: question_events logs every question without IP storage, so hosted deploy can keep the same privacy rule.
- Environment variables: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT, GROQ_API_KEY, TAVILY_API_KEY and GROQ_MODEL from config.py and .env.example. CI must inject these as secrets and never log them.
- Admin scrape: POST /admin/operations/scrape enforces a single concurrent run with a lock plus 60 second cooldown, records admin:<email> as triggered_by and polls via GET /admin/operations/scrape/{job_id}.
- Freshness view: area_freshness shows listing count and last ingested time per area for monitoring hosted ingest.

Suggested CI shape (not built in this task):

1. On schedule (for example nightly at 02:00 Africa/Lagos), checkout on Python 3.12, run uv sync.
2. Run migrations with uv run python -m entrypoints.cli migrate against hosted Postgres.
3. Run uv run python -c "from scraping.runner import run_scrape; run_scrape(triggered_by='ci-schedule')".
4. Run pipeline/load_data.py against the fresh CSV so dedup and area_freshness update.
5. Upload quality report and log_excerpt as CI artifacts, then alert on scraper_runs status failed or error_count above zero.

Decisions needed from owner:

- Hosted Postgres provider and region, and who owns backups and rotation.
- Scrape frequency and max pages per run to respect site rate limits.
- Whether advice traffic should keep using live Tavily and Groq keys in CI or move to nightly scheduled eval only.
