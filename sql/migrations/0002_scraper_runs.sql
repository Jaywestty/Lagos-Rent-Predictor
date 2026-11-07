CREATE TABLE IF NOT EXISTS scraper_runs (
    id SERIAL PRIMARY KEY,
    triggered_by TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    listings_discovered INTEGER DEFAULT 0,
    listings_added INTEGER DEFAULT 0,
    listings_updated INTEGER DEFAULT 0,
    duplicates_skipped INTEGER DEFAULT 0,
    validation_failures INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    log_excerpt TEXT
);
CREATE INDEX IF NOT EXISTS idx_scraper_runs_started_at ON scraper_runs(started_at);
