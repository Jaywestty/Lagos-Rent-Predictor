-- Chat history: one thread per conversation, owned by an anonymous per-device client id.
-- The thread id is also the session id used for follow-up memory.
CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    title TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_threads_client ON chat_threads (client_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    payload JSONB NOT NULL,            -- answer card, matching listings, or advice with sources
    request_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages (thread_id, id);

ALTER TABLE chat_threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
