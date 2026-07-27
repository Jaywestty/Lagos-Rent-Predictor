import os
from functools import lru_cache

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    user = os.getenv("POSTGRES_USER", "lagos_rent")
    password = os.getenv("POSTGRES_PASSWORD", "lagos_rent")
    db = os.getenv("POSTGRES_DB", "lagos_rent")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    logger.info("Creating database engine for host={} db={}", host, db)
    return create_engine(url, pool_pre_ping=True)