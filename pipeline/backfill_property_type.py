import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from lagos_property_scraper import parse_npc_url

logger.add("logs/backfill.log", rotation="10 MB", retention="14 days", level="INFO")


def backfill_npc_property_type(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "source" not in df.columns or "property_type" not in df.columns:
        raise ValueError("CSV is missing expected columns: source, property_type")

    npc_mask = df["source"] == "npc"
    npc_count = int(npc_mask.sum())
    logger.info("Found {} npc rows out of {} total rows", npc_count, len(df))

    changed_count = 0
    for idx in df[npc_mask].index:
        url = df.at[idx, "url"]
        old_value = df.at[idx, "property_type"]
        parsed = parse_npc_url(url)
        new_value = parsed.get("property_type")

        if new_value and new_value != old_value:
            df.at[idx, "property_type"] = new_value
            changed_count += 1
            logger.info("Corrected property_type for {}: '{}' -> '{}'", url, old_value, new_value)

    logger.info("Backfill complete. {} rows corrected.", changed_count)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/lagos_properties.csv")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would change without writing the file.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = backfill_npc_property_type(csv_path)

    if args.dry_run:
        logger.info("Dry run — no file written.")
        return

    backup_path = csv_path.with_suffix(".csv.bak")
    csv_path.rename(backup_path)
    logger.info("Original CSV backed up to {}", backup_path)

    df.to_csv(csv_path, index=False)
    logger.info("Corrected CSV written to {}", csv_path)


if __name__ == "__main__":
    main()