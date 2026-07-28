import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
url = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'lagos_rent')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'lagos_rent')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'lagos_rent')}"
)
engine = create_engine(url)

with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT a.id, a.title, a.price_ngn, a.area, a.subarea,
               b.id AS dup_id, b.title AS dup_title, b.price_ngn AS dup_price
        FROM listings a
        JOIN listings b ON b.is_duplicate_of = a.id
        LIMIT 10
    """)).fetchall()
    for r in rows:
        print(r)