import os
from functools import lru_cache

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url

load_dotenv()


def _database_url() -> str:
    """Return the SQLAlchemy URL.

    DATABASE_URL (for example the Supabase connection string) wins. The
    POSTGRES_* variables remain as a fallback for a local Postgres.
    """
    raw = os.getenv("DATABASE_URL")
    if raw:
        if raw.startswith("postgres://"):
            raw = "postgresql://" + raw[len("postgres://"):]
        if raw.startswith("postgresql://"):
            raw = "postgresql+psycopg2://" + raw[len("postgresql://"):]
        return raw
    user = os.getenv("POSTGRES_USER", "lagos_rent")
    password = os.getenv("POSTGRES_PASSWORD", "lagos_rent")
    db = os.getenv("POSTGRES_DB", "lagos_rent")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = make_url(_database_url())
    connect_args: dict[str, str] = {}
    host = url.host or ""
    # Hosted Postgres (Supabase and similar) requires TLS.
    if host not in ("localhost", "127.0.0.1", "::1") and "sslmode" not in url.query:
        connect_args["sslmode"] = "require"
    logger.info("Creating database engine for host={} db={}", host, url.database)
    return create_engine(url, pool_pre_ping=True, pool_recycle=300, connect_args=connect_args)
