-- Set by the admin "Stop scrape" button; the running scraper checks it and stops cleanly.
ALTER TABLE scraper_runs ADD COLUMN IF NOT EXISTS stop_requested_at TIMESTAMPTZ;
