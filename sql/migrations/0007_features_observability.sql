-- Weekly price snapshots per area and bucket, written after every scrape run.
CREATE TABLE IF NOT EXISTS price_snapshots (
    id SERIAL PRIMARY KEY,
    week_start DATE NOT NULL,
    area TEXT NOT NULL,
    bucket TEXT NOT NULL,           -- all, self_contain, one_bedroom, two_bedroom, three_bedroom
    listing_count INTEGER NOT NULL,
    median_ngn NUMERIC,
    p10_ngn NUMERIC,
    p90_ngn NUMERIC,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (week_start, area, bucket)
);
CREATE INDEX IF NOT EXISTS idx_price_snapshots_area ON price_snapshots (area, bucket, week_start);

-- Cached area guides built from web search (Tavily) plus Groq, with sources.
CREATE TABLE IF NOT EXISTS area_guides (
    area TEXT PRIMARY KEY,
    content JSONB NOT NULL,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    model TEXT,
    generated_at TIMESTAMPTZ DEFAULT now()
);

-- Anonymous per-device saved searches and favourites (client_id is a random UUID kept in the browser).
CREATE TABLE IF NOT EXISTS saved_searches (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    label TEXT NOT NULL,
    query TEXT,
    area TEXT,
    bedrooms INTEGER,
    max_budget_ngn NUMERIC,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_saved_searches_client ON saved_searches (client_id);

CREATE TABLE IF NOT EXISTS favorites (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    price_text TEXT,
    annual_price_ngn NUMERIC,
    area TEXT,
    bedrooms INTEGER,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (client_id, url)
);
CREATE INDEX IF NOT EXISTS idx_favorites_client ON favorites (client_id);

-- Every HTTP request the API serves (no IP, no body).
CREATE TABLE IF NOT EXISTS api_requests (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    method TEXT NOT NULL,
    route TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    duration_ms NUMERIC NOT NULL,
    client_id TEXT,
    user_agent_family TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_api_requests_created ON api_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_api_requests_route ON api_requests (route, created_at);

-- Unhandled exceptions and handled failures worth seeing (LLM, search, DB).
CREATE TABLE IF NOT EXISTS error_events (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    source TEXT NOT NULL,            -- http, groq, tavily, scraper, guide, db
    route TEXT,
    error_type TEXT,
    message TEXT,
    stack TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_error_events_created ON error_events (created_at);

-- Every external AI / search call: provider, purpose, latency, tokens, outcome.
CREATE TABLE IF NOT EXISTS llm_calls (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT,
    provider TEXT NOT NULL,          -- groq, tavily
    purpose TEXT NOT NULL,           -- extraction, advice, guide, search
    model TEXT,
    latency_ms NUMERIC NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    success BOOLEAN NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls (created_at);
ALTER TABLE llm_calls ENABLE ROW LEVEL SECURITY;

-- Product usage events (guide views, compares, calculator use, saves).
CREATE TABLE IF NOT EXISTS feature_events (
    id BIGSERIAL PRIMARY KEY,
    client_id TEXT,
    feature TEXT NOT NULL,
    detail JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feature_events_created ON feature_events (feature, created_at);

ALTER TABLE price_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE area_guides ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_searches ENABLE ROW LEVEL SECURITY;
ALTER TABLE favorites ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE error_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_events ENABLE ROW LEVEL SECURITY;
