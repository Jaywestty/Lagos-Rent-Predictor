import os

# Tests must never touch the real (Supabase) database. Point the engine at a closed local
# port before anything imports db.py; load_dotenv() does not override variables already set.
os.environ["DATABASE_URL"] = "postgresql://test:test@127.0.0.1:1/test"

from observability import events  # noqa: E402

# And never queue observability rows.
events.disable()
