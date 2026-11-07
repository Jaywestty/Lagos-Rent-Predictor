CREATE TABLE IF NOT EXISTS question_events (
    id SERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    raw_question TEXT,
    parsed_location TEXT,
    parsed_property_type TEXT,
    status TEXT,
    listing_count INTEGER,
    confidence TEXT,
    latency_ms NUMERIC,
    time_to_first_event_ms NUMERIC,
    error_code TEXT,
    review_state TEXT DEFAULT 'pending',
    review_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_question_events_created_at ON question_events(created_at);
CREATE INDEX IF NOT EXISTS idx_question_events_request_id ON question_events(request_id);
