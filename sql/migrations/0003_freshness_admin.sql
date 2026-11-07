CREATE OR REPLACE VIEW area_freshness AS
SELECT area AS area, COUNT(*) AS listing_count, MAX(COALESCE(scraped_at, created_at)) AS last_ingested_at
FROM listings
WHERE is_duplicate_of IS NULL
GROUP BY area;

CREATE TABLE IF NOT EXISTS admin_users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    created_at TIMESTAMPTZ DEFAULT now(),
    last_login_at TIMESTAMPTZ
);
