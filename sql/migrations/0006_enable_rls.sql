-- Supabase exposes every table in the public schema through its REST API.
-- The backend connects as the table owner, which bypasses RLS, so enabling RLS
-- with no policies blocks anon/authenticated REST access without affecting the app.
ALTER TABLE listings ENABLE ROW LEVEL SECURITY;
ALTER TABLE location_aliases ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE scraper_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE login_attempts ENABLE ROW LEVEL SECURITY;

-- Views run with the owner's rights by default; make this one respect RLS of the caller.
ALTER VIEW area_freshness SET (security_invoker = true);
