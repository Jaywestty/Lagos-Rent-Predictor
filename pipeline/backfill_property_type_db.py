import argparse

from loguru import logger
from sqlalchemy import text

from db import get_engine
from lagos_property_scraper import parse_npc_url

logger.add("logs/backfill_db.log", rotation="10 MB", retention="14 days", level="INFO")


def fetch_bad_rows(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, url FROM listings WHERE source = 'npc' AND property_type = 'Lagos'")
        ).mappings().all()
    return [dict(row) for row in rows]


def backfill(dry_run: bool):
    engine = get_engine()
    bad_rows = fetch_bad_rows(engine)
    logger.info("Found {} rows with property_type = 'Lagos'", len(bad_rows))

    updates = []
    unresolved = []
    for row in bad_rows:
        parsed = parse_npc_url(row["url"])
        new_value = parsed.get("property_type")
        if new_value and new_value != "Lagos":
            updates.append({"id": row["id"], "property_type": new_value})
            logger.info("Row {}: '{}' -> '{}' ({})", row["id"], "Lagos", new_value, row["url"])
        else:
            unresolved.append(row["url"])
            logger.warning("Row {} could not be resolved from URL: {}", row["id"], row["url"])

    logger.info("Resolved: {}. Unresolved: {}.", len(updates), len(unresolved))

    if dry_run:
        logger.info("Dry run — no database changes made.")
        return

    stmt = text("UPDATE listings SET property_type = :property_type WHERE id = :id")
    with engine.begin() as conn:
        for params in updates:
            conn.execute(stmt, params)

    logger.info("Updated {} rows in the database.", len(updates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would change without writing to the database.")
    args = parser.parse_args()
    backfill(args.dry_run)


if __name__ == "__main__":
    main()