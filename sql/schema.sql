CREATE TABLE IF NOT EXISTS listings (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    listing_type TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    price_raw TEXT,
    price_ngn NUMERIC,
    price_period TEXT,
    location_raw TEXT,
    area TEXT,
    subarea TEXT,
    bedrooms INTEGER,
    bathrooms INTEGER,
    toilets INTEGER,
    parking INTEGER,
    property_type TEXT,
    is_duplicate_of INTEGER REFERENCES listings(id),
    price_anomaly_flag BOOLEAN DEFAULT FALSE,
    scraped_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS location_aliases (
    raw_variant TEXT PRIMARY KEY,
    normalized_area TEXT NOT NULL,
    normalized_subarea TEXT,
    needs_review BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_listings_area_beds ON listings (area, bedrooms, listing_type);
CREATE INDEX IF NOT EXISTS idx_listings_price ON listings (price_ngn);
CREATE INDEX IF NOT EXISTS idx_listings_url ON listings (url);