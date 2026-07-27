from typing import Optional

from entities import PropertyEntities


def format_price(price_ngn: Optional[float], price_period: Optional[str]) -> str:
    if price_ngn is None:
        return "price not stated"
    period_label = f"/{price_period}" if price_period else ""
    return f"NGN {price_ngn:,.0f}{period_label}"


def build_lookup_response(entities: PropertyEntities, listings: list[dict]) -> dict:
    if not listings:
        return {
            "query_type": entities.query_type.value,
            "matched_count": 0,
            "message": "No matching listings found for that search.",
            "results": [],
        }

    results = [
        {
            "title": listing["title"],
            "price": format_price(listing["price_ngn"], listing["price_period"]),
            "area": listing["area"],
            "subarea": listing["subarea"],
            "bedrooms": listing["bedrooms"],
            "url": listing["url"],
        }
        for listing in listings
    ]

    return {
        "query_type": entities.query_type.value,
        "matched_count": len(results),
        "message": f"Found {len(results)} matching listing(s).",
        "results": results,
    }