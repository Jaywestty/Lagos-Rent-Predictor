"""Area-level market numbers: area list, area stats, weekly trends, comparisons.

Every figure comes from non-duplicate rent listings with a positive price,
annualised the same way as the chat answer (monthly x 12, null treated as yearly).
"""

from collections import defaultdict
from datetime import date, datetime, timedelta

from sqlalchemy import text

from config import LIMITED_CONFIDENCE_MAX, LOW_DATA_MAX, PERCENTILE_MIN_COUNT
from db import get_engine
from query.stats import annualized_price, bucket_for, classify, summarize_values
from scraping.location_normalizer import normalize_location

BUCKETS = ("self_contain", "one_bedroom", "two_bedroom", "three_bedroom")
BUCKET_LABELS = {
    "all": "All listings",
    "self_contain": "Self contain",
    "one_bedroom": "1 bedroom",
    "two_bedroom": "2 bedroom",
    "three_bedroom": "3 bedroom",
}
BEDS_TO_BUCKET = {1: "one_bedroom", 2: "two_bedroom", 3: "three_bedroom"}

_BASE_WHERE = "is_duplicate_of IS NULL AND listing_type = 'rent' AND price_ngn IS NOT NULL AND price_ngn > 0 AND area IS NOT NULL"


def canonical_area(area: str | None) -> str | None:
    if not area:
        return None
    try:
        return normalize_location(area).get("area") or area.strip().title()
    except Exception:
        return area.strip().title()


def _rows(area: str | None = None) -> list[dict]:
    sql = f"SELECT area, bedrooms, property_type, price_ngn, price_period, listed_at, scraped_at, created_at FROM listings WHERE {_BASE_WHERE}"
    params = {}
    if area:
        sql += " AND lower(area) = lower(:area)"
        params["area"] = area
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


def _summarize(values: list[float]) -> dict | None:
    return summarize_values(values, PERCENTILE_MIN_COUNT) if values else None


def _latest(rows: list[dict]) -> str | None:
    stamps: list[datetime] = [s for s in (r.get("listed_at") or r.get("scraped_at") or r.get("created_at") for r in rows) if s]
    return max(stamps).isoformat() if stamps else None


def list_areas(min_count: int = 1) -> list[dict]:
    """Every area with listings, its count, overall median and newest listing date."""
    by_area: dict[str, list[dict]] = defaultdict(list)
    for r in _rows():
        by_area[r["area"]].append(r)
    out = []
    for area, rows in by_area.items():
        values = [v for v in (annualized_price(r["price_ngn"], r["price_period"]) for r in rows) if v]
        if len(values) < min_count:
            continue
        s = _summarize(values)
        out.append({
            "area": area,
            "listing_count": len(values),
            "median_ngn": s["median"] if s else None,
            "latest_listing_at": _latest(rows),
        })
    out.sort(key=lambda a: (-int(a["listing_count"] or 0), str(a["area"])))
    return out


def area_stats(area: str) -> dict:
    """Overall and per-bucket stats for one area, with confidence."""
    canon = canonical_area(area)
    rows = _rows(canon)
    all_values: list[float] = []
    buckets: dict[str, list[float]] = {b: [] for b in BUCKETS}
    for r in rows:
        v = annualized_price(r["price_ngn"], r["price_period"])
        if v is None:
            continue
        all_values.append(v)
        b = bucket_for(r["bedrooms"], r["property_type"])
        if b in buckets:
            buckets[b].append(v)
    status, confidence = classify(len(all_values), LOW_DATA_MAX, LIMITED_CONFIDENCE_MAX)
    return {
        "area": rows[0]["area"] if rows else canon,
        "listing_count": len(all_values),
        "status": status,
        "confidence": confidence,
        "overall": _summarize(all_values),
        "breakdown": {b: _summarize(vals) for b, vals in buckets.items()},
        "data_as_of": _latest(rows),
    }


def _week_start(d) -> date:
    if isinstance(d, datetime):
        d = d.date()
    return d - timedelta(days=d.weekday())


def _percentile(sorted_vals: list[float], pct: float) -> float:
    from query.stats import _percentile as p

    return p(sorted_vals, pct)


def write_snapshots() -> int:
    """Recompute weekly snapshots (median, p10, p90 per area and bucket) from listing dates.

    A listing belongs to the week of its source date (listed_at), falling back to
    when it was scraped. Re-running is safe: each (week, area, bucket) is upserted.
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in _rows():
        when = r.get("listed_at") or r.get("scraped_at") or r.get("created_at")
        v = annualized_price(r["price_ngn"], r["price_period"])
        if not when or v is None:
            continue
        week = _week_start(when)
        groups[(week, r["area"], "all")].append(v)
        b = bucket_for(r["bedrooms"], r["property_type"])
        if b:
            groups[(week, r["area"], b)].append(v)
    rows = []
    for (week, area, bucket), vals in groups.items():
        vals.sort()
        rows.append({
            "week_start": week, "area": area, "bucket": bucket, "listing_count": len(vals),
            "median_ngn": summarize_values(vals)["median"],
            "p10_ngn": round(_percentile(vals, 10), 2), "p90_ngn": round(_percentile(vals, 90), 2),
        })
    if not rows:
        return 0
    with get_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO price_snapshots (week_start, area, bucket, listing_count, median_ngn, p10_ngn, p90_ngn)
            VALUES (:week_start, :area, :bucket, :listing_count, :median_ngn, :p10_ngn, :p90_ngn)
            ON CONFLICT (week_start, area, bucket) DO UPDATE SET
              listing_count = EXCLUDED.listing_count, median_ngn = EXCLUDED.median_ngn,
              p10_ngn = EXCLUDED.p10_ngn, p90_ngn = EXCLUDED.p90_ngn, created_at = now()
        """), rows)
    return len(rows)


def trends(area: str, bucket: str = "all", weeks: int = 26) -> dict:
    """Weekly median series for one area and bucket, oldest first, plus change vs first point."""
    canon = canonical_area(area)
    if bucket not in BUCKET_LABELS:
        bucket = "all"
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT week_start, listing_count, median_ngn, p10_ngn, p90_ngn FROM price_snapshots
            WHERE lower(area) = lower(:area) AND bucket = :bucket AND week_start >= :since
            ORDER BY week_start
        """), {"area": canon, "bucket": bucket, "since": date.today() - timedelta(weeks=weeks)}).mappings().all()
    points = [{
        "week_start": r["week_start"].isoformat(),
        "listing_count": r["listing_count"],
        "median_ngn": float(r["median_ngn"]) if r["median_ngn"] is not None else None,
        "p10_ngn": float(r["p10_ngn"]) if r["p10_ngn"] is not None else None,
        "p90_ngn": float(r["p90_ngn"]) if r["p90_ngn"] is not None else None,
    } for r in rows]
    change = None
    usable = [p for p in points if p["median_ngn"] and p["listing_count"] >= 3]
    if len(usable) >= 2:
        first, last = usable[0]["median_ngn"], usable[-1]["median_ngn"]
        change = round((last - first) / first * 100, 1)
    return {"area": canon, "bucket": bucket, "label": BUCKET_LABELS[bucket], "points": points,
            "change_pct": change, "weeks_with_data": len(points)}


def compare(areas: list[str], beds: int | None = None) -> dict:
    """Side-by-side stats for up to three areas, optionally for one bedroom bucket."""
    bucket = BEDS_TO_BUCKET.get(beds) if beds is not None else None
    out = []
    for a in areas[:3]:
        stats = area_stats(a)
        focus = stats["breakdown"].get(bucket) if bucket else stats["overall"]
        trend = trends(stats["area"] or a, bucket or "all")
        out.append({**stats, "focus": focus, "trend": trend})
    return {"bucket": bucket or "all", "label": BUCKET_LABELS[bucket or "all"], "areas": out}
