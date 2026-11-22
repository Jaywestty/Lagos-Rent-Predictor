-- Live progress of a running scrape (current site/category/page, counts, heartbeat), shown in admin.
ALTER TABLE scraper_runs ADD COLUMN IF NOT EXISTS progress JSONB;
