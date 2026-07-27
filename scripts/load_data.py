import argparse
import json
import os
from pathlib import Path

from numpy.ma import anomalies
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import Table, MetaData

from db import get_engine

from location_normalizer import normalize_location

DEDUP_SUBAREA_SIMILARITY_THRESHOLD = 85
DEDUP_PRICE_TOLERANCE_PCT = 0.05

DEDUP_TITLE_SIMILARITY_THRESHOLD = 60
DEDUP_STRICT_TITLE_SIMILARITY_THRESHOLD = 80

DEDUP_HIGH_CONFIDENCE_TITLE_THRESHOLD = 80

RENT_MONTHLY_SANITY_CEILING_NGN = 10_000_000
GENERIC_PROPERTY_TYPE_VALUES = {"lagos"}
RENT_ANNUAL_SANITY_FLOOR_NGN = 300_000


def apply_schema(engine, schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with engine.begin() as conn:
        conn.execute(text(schema_sql))


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"location": "location_raw"})

    for col in ["bedrooms", "bathrooms", "toilets", "parking"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["price_ngn"] = pd.to_numeric(df["price_ngn"], errors="coerce")
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["price_period"] = df["price_period"].where(df["price_period"].notna(), None)

    return df


def apply_location_normalization(df):
    normalized = df["location_raw"].apply(normalize_location)
    df["area"] = normalized.apply(lambda r: r["area"])
    df["subarea"] = normalized.apply(lambda r: r["subarea"])
    df["_location_needs_review"] = normalized.apply(lambda r: r["needs_review"])
    return df


def flag_dedup_groups(df):
    df = df.reset_index(drop=True)
    df["is_duplicate_of_row"] = None
    df["dedup_review_flag"] = False

    group_cols = ["bedrooms", "area", "property_type", "listing_type"]
    for _, group in df.groupby(group_cols, dropna=False):
        if len(group) < 2:
            continue
        indices = list(group.index)
        for i in range(len(indices)):
            idx_a = indices[i]
            if df.loc[idx_a, "is_duplicate_of_row"] is not None:
                continue
            for j in range(i + 1, len(indices)):
                idx_b = indices[j]
                if df.loc[idx_b, "is_duplicate_of_row"] is not None:
                    continue

                subarea_a = df.loc[idx_a, "subarea"]
                subarea_b = df.loc[idx_b, "subarea"]

                price_a = df.loc[idx_a, "price_ngn"]
                price_b = df.loc[idx_b, "price_ngn"]
                if pd.isna(price_a) or pd.isna(price_b):
                    continue
                price_diff_pct = abs(price_a - price_b) / max(price_a, price_b)

                if price_diff_pct > DEDUP_PRICE_TOLERANCE_PCT:
                    continue

                title_a = df.loc[idx_a, "title"] or ""
                title_b = df.loc[idx_b, "title"] or ""
                title_similarity = fuzz.token_sort_ratio(title_a, title_b)

                subarea_missing = pd.isna(subarea_a) or pd.isna(subarea_b)
                if not subarea_missing:
                    subarea_similarity = fuzz.ratio(subarea_a, subarea_b)
                    if subarea_similarity < DEDUP_SUBAREA_SIMILARITY_THRESHOLD:
                        continue

                if title_similarity >= DEDUP_HIGH_CONFIDENCE_TITLE_THRESHOLD:
                    df.loc[idx_b, "is_duplicate_of_row"] = idx_a
                elif title_similarity >= DEDUP_TITLE_SIMILARITY_THRESHOLD:
                    df.loc[idx_b, "dedup_review_flag"] = True
                    df.loc[idx_a, "dedup_review_flag"] = True

    return df


def flag_load_time_anomalies(df):
    anomalies = {}

    bad_property_type = df[df["property_type"].str.lower().isin(GENERIC_PROPERTY_TYPE_VALUES)]
    anomalies["property_type_equals_area_bug"] = bad_property_type["url"].tolist()

    missing_period_rent = df[(df["listing_type"] == "rent") & (df["price_period"].isna())]
    anomalies["rent_missing_price_period"] = missing_period_rent["url"].tolist()

    implausible_monthly_rent = df[
        (df["listing_type"] == "rent")
        & (df["price_period"] == "month")
        & (df["price_ngn"] > RENT_MONTHLY_SANITY_CEILING_NGN)
    ]
    anomalies["implausible_monthly_rent"] = implausible_monthly_rent["url"].tolist()

    implausible_annual_rent = df[
    (df["listing_type"] == "rent")
    & (df["price_period"] == "year")
    & (df["price_ngn"] < RENT_ANNUAL_SANITY_FLOOR_NGN)]
    anomalies["implausible_annual_rent"] = implausible_annual_rent["url"].tolist()

    missing_location = df[df["location_raw"].isna()]
    anomalies["missing_location"] = missing_location["url"].tolist()

    return anomalies


def build_quality_report(df, anomalies, dedup_count, review_needed_count, dedup_review_count):
    report = {
        "total_rows": len(df),
        "duplicate_rows_flagged": dedup_count,
        "locations_needing_manual_review": review_needed_count,
        "possible_duplicates_needing_manual_review": dedup_review_count,
        "nulls_per_column": df.isna().sum().to_dict(),
        "price_distribution_by_area_bedrooms": (
            df.groupby(["area", "bedrooms"])["price_ngn"]
            .describe()[["count", "mean", "min", "max"]]
            .reset_index()
            .to_dict(orient="records")
        ),
        "anomalies": anomalies,
    }
    return report


def upsert_listings(df, engine):
    metadata = MetaData()
    listings_table = Table("listings", metadata, autoload_with=engine)

    records = df.drop(columns=["_location_needs_review", "is_duplicate_of_row"], errors="ignore")
    records = records.where(pd.notna(records), None)
    rows = records.to_dict(orient="records")

    with engine.begin() as conn:
        for row in rows:
            stmt = pg_insert(listings_table).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["url"],
                set_={k: v for k, v in row.items() if k != "url"},
            )
            conn.execute(stmt)


def upsert_location_aliases(df, engine):
    metadata = MetaData()
    aliases_table = Table("location_aliases", metadata, autoload_with=engine)

    review_rows = df[df["_location_needs_review"] == True]
    seen = set()
    skipped_no_location = 0
    with engine.begin() as conn:
        for _, row in review_rows.iterrows():
            raw = row["location_raw"]
            if pd.isna(raw) or row["area"] is None:
                skipped_no_location += 1
                continue
            if raw in seen:
                continue
            seen.add(raw)
            stmt = pg_insert(aliases_table).values(
                raw_variant=raw,
                normalized_area=row["area"],
                normalized_subarea=row["subarea"],
                needs_review=True,
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=["raw_variant"])
            conn.execute(stmt)

def apply_dedup_to_db(df, engine):
    urls_in_scope = df["url"].tolist()

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE listings SET is_duplicate_of = NULL WHERE url = ANY(:urls)"),
            {"urls": urls_in_scope},
        )

    dup_pairs = df[df["is_duplicate_of_row"].notna()]
    if dup_pairs.empty:
        return 0

    updates = []
    for dup_idx, row in dup_pairs.iterrows():
        orig_idx = row["is_duplicate_of_row"]
        updates.append({
            "dup_url": row["url"],
            "orig_url": df.loc[orig_idx, "url"],
        })

    stmt = text("""
        UPDATE listings AS l
        SET is_duplicate_of = orig.id
        FROM listings AS orig
        WHERE l.url = :dup_url AND orig.url = :orig_url
    """)

    with engine.begin() as conn:
        for params in updates:
            conn.execute(stmt, params)

    return len(updates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/lagos_properties.csv")
    parser.add_argument("--schema", default="sql/schema.sql")
    parser.add_argument("--report-out", default="data/quality_report.json")
    args = parser.parse_args()

    engine = get_engine()
    apply_schema(engine, args.schema)

    df = load_csv(args.csv)
    df = apply_location_normalization(df)
    df = flag_dedup_groups(df)

    dedup_count = df["is_duplicate_of_row"].notna().sum()
    dedup_review_count = int(df["dedup_review_flag"].sum())
    review_needed_count = int(df["_location_needs_review"].sum())
    anomalies = flag_load_time_anomalies(df)

    report = build_quality_report(df, anomalies, int(dedup_count), review_needed_count, dedup_review_count)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    df_for_insert = df.drop(columns=["is_duplicate_of_row", "dedup_review_flag"])
    upsert_listings(df_for_insert, engine)
    upsert_location_aliases(df, engine)
    apply_dedup_to_db(df, engine)

    print(f"Loaded {len(df)} rows. Duplicates flagged: {dedup_count}. Needs review: {review_needed_count}.")
    print(f"Quality report written to {args.report_out}")


if __name__ == "__main__":
    main()