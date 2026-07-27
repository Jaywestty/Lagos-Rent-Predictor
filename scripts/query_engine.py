from typing import Any

from loguru import logger
from sqlalchemy import text

from db import get_engine
from entities import PropertyEntities, QueryType
from location_normalizer import normalize_location

LOOKUP_RESULT_LIMIT = 10


def build_lookup_query(entities: PropertyEntities):
    filters = ["is_duplicate_of IS NULL"]
    params: dict[str, Any] = {"limit": LOOKUP_RESULT_LIMIT}

    if entities.area:
        normalized = normalize_location(entities.area)
        resolved_area = normalized.get("area") or entities.area
        filters.append("area = :area")
        params["area"] = resolved_area

    if entities.beds is not None:
        filters.append("bedrooms = :beds")
        params["beds"] = entities.beds

    if entities.listing_type is not None:
        filters.append("listing_type = :listing_type")
        params["listing_type"] = entities.listing_type.value

    where_clause = " AND ".join(filters)
    sql = text(f"""
        SELECT id, source, listing_type, url, title, price_ngn, price_period,
               area, subarea, bedrooms, bathrooms, toilets, parking, property_type
        FROM listings
        WHERE {where_clause}
        ORDER BY (CASE WHEN price_period = 'month' THEN price_ngn * 12 ELSE price_ngn END) ASC NULLS LAST
        LIMIT :limit
    """)
    return sql, params

def run_lookup(entities: PropertyEntities) -> list[dict]:
    if entities.query_type != QueryType.LOOKUP:
        raise ValueError(f"run_lookup called with non-lookup query_type: {entities.query_type}")

    if entities.area is None and entities.beds is None and entities.listing_type is None and entities.property_type is None:
        logger.warning("Lookup query has no usable filters, refusing to run an unfiltered scan")
        return []
    stmt, params = build_lookup_query(entities)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    logger.info(
        "Lookup returned {} rows for area={} beds={} listing_type={}",
        len(rows), entities.area, entities.beds, entities.listing_type,
    )
    return [dict(row) for row in rows]