-- When the source site says the listing was posted or last updated.
-- NigeriaPropertyCentre: JSON-LD datePosted. PropertyPro: "Last updated" on the detail page.
ALTER TABLE listings ADD COLUMN IF NOT EXISTS listed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_listings_listed_at ON listings (listed_at);

-- Weekly runs skip listings older than the recency window; record how many.
ALTER TABLE scraper_runs ADD COLUMN IF NOT EXISTS listings_too_old INTEGER DEFAULT 0;
ALTER TABLE scraper_runs ADD COLUMN IF NOT EXISTS since_days INTEGER;
