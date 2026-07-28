from typing import Any

from loguru import logger
from sqlalchemy import text

from db import get_engine
from query.entities import PropertyEntities, QueryType
from scraping.location_normalizer import normalize_location

LOOKUP_RESULT_LIMIT = 10
AFFORDABILITY_RESULT_LIMIT = 30
AFFORDABILITY_STRETCH_FACTOR = 1.15
COMPARISON_RESULT_LIMIT = 5
SPARSE_AREA_THRESHOLD = 3


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

    if entities.listing_type is not None:
        filters.append("listing_type = :listing_type")
        params["listing_type"] = entities.listing_type.value
    else:
        filters.append("listing_type = 'rent'")

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

def _annualized_price_expr() -> str:
    return "(CASE WHEN price_period = 'month' THEN price_ngn * 12 ELSE price_ngn END)"


def build_affordability_query(entities: PropertyEntities):
    annual_price = _annualized_price_expr()
    filters = ["is_duplicate_of IS NULL", "price_ngn IS NOT NULL"]
    params: dict[str, Any] = {
        "stretch_budget": entities.budget_ngn * AFFORDABILITY_STRETCH_FACTOR,
        "limit": AFFORDABILITY_RESULT_LIMIT,
    }
    filters.append(f"{annual_price} <= :stretch_budget")

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

    if entities.property_type:
        filters.append("property_type = :property_type")
        params["property_type"] = entities.property_type

    where_clause = " AND ".join(filters)
    sql = text(f"""
        SELECT id, source, listing_type, url, title, price_ngn, price_period,
               area, subarea, bedrooms, bathrooms, toilets, parking, property_type,
               {annual_price} AS annual_price
        FROM listings
        WHERE {where_clause}
        ORDER BY annual_price ASC
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

def run_affordability(entities: PropertyEntities) -> dict:
    if entities.query_type != QueryType.AFFORDABILITY:
        raise ValueError(f"run_affordability called with non-affordability query_type: {entities.query_type}")

    if entities.budget_ngn is None:
        logger.warning("Affordability query has no budget, refusing to run")
        return {"within_budget": {}, "stretch": {}, "total_matches": 0}

    stmt, params = build_affordability_query(entities)
    engine = get_engine()
    with engine.connect() as conn:
        rows = [dict(row) for row in conn.execute(stmt, params).mappings().all()]

    within_budget = [r for r in rows if r["annual_price"] <= entities.budget_ngn]
    stretch = [r for r in rows if r["annual_price"] > entities.budget_ngn]

    grouped_within: dict[str, list[dict]] = {}
    for row in within_budget:
        grouped_within.setdefault(row["area"] or "unspecified", []).append(row)

    grouped_stretch: dict[str, list[dict]] = {}
    for row in stretch:
        grouped_stretch.setdefault(row["area"] or "unspecified", []).append(row)

    logger.info(
        "Affordability query returned {} within budget, {} in stretch band for budget={}",
        len(within_budget), len(stretch), entities.budget_ngn,
    )
    return {
        "within_budget": grouped_within,
        "stretch": grouped_stretch,
        "total_matches": len(within_budget),
    }


def run_comparison(entities: PropertyEntities) -> dict:
    if entities.query_type != QueryType.COMPARISON:
        raise ValueError(f"run_comparison called with non-comparison query_type: {entities.query_type}")

    if not entities.comparison_options or len(entities.comparison_options) < 2:
        logger.warning("Comparison query has fewer than 2 options, refusing to run")
        return {}

    results: dict[str, list[dict]] = {}
    engine = get_engine()

    for option in entities.comparison_options:
        sub_entities = PropertyEntities(
            query_type=QueryType.LOOKUP,
            beds=option.beds,
            area=option.area,
            property_type=option.property_type,
            listing_type=option.listing_type,
        )
        if (
            sub_entities.area is None
            and sub_entities.beds is None
            and sub_entities.listing_type is None
            and sub_entities.property_type is None
        ):
            logger.warning("Comparison option '{}' has no usable filters, skipping", option.label)
            results[option.label] = []
            continue

        stmt, params = build_lookup_query(sub_entities)
        params["limit"] = COMPARISON_RESULT_LIMIT
        with engine.connect() as conn:
            rows = conn.execute(stmt, params).mappings().all()
        results[option.label] = [dict(row) for row in rows]

    return results