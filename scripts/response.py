from typing import Optional

from entities import PropertyEntities
from query_engine import SPARSE_AREA_THRESHOLD


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


def build_affordability_response(entities: PropertyEntities, result: dict) -> dict:
    within_budget = result["within_budget"]
    stretch = result["stretch"]
    total_matches = result["total_matches"]

    if not within_budget and not stretch:
        return {
            "query_type": entities.query_type.value,
            "budget_ngn": entities.budget_ngn,
            "matched_count": 0,
            "message": "No listings found near that budget.",
            "areas": [],
            "stretch_options": [],
        }

    areas = []
    for area_name, rows in sorted(within_budget.items(), key=lambda kv: min(r["annual_price"] for r in kv[1])):
        areas.append({
            "area": area_name,
            "matched_count": len(rows),
            "sparse": len(rows) < SPARSE_AREA_THRESHOLD,
            "results": [
                {
                    "title": r["title"],
                    "price": format_price(r["price_ngn"], r["price_period"]),
                    "bedrooms": r["bedrooms"],
                    "url": r["url"],
                }
                for r in rows[:5]
            ],
        })

    stretch_sorted = sorted(stretch.items(), key=lambda kv: min(r["annual_price"] for r in kv[1]))[:5]
    stretch_areas = [
        {
            "area": area_name,
            "results": [
                {
                    "title": r["title"],
                    "price": format_price(r["price_ngn"], r["price_period"]),
                    "bedrooms": r["bedrooms"],
                    "url": r["url"],
                }
                for r in rows[:3]
            ],
        }
        for area_name, rows in stretch_sorted
    ]

    message = f"Found {total_matches} listing(s) within budget across {len(within_budget)} area(s)."
    if not within_budget and stretch:
        message = "Nothing found within budget, but here are close options slightly above it."

    return {
        "query_type": entities.query_type.value,
        "budget_ngn": entities.budget_ngn,
        "matched_count": total_matches,
        "message": message,
        "areas": areas,
        "stretch_options": stretch_areas,
    }


def build_comparison_response(entities: PropertyEntities, results_by_option: dict) -> dict:
    if not results_by_option:
        return {
            "query_type": entities.query_type.value,
            "message": "Could not compare — need at least 2 valid options.",
            "options": [],
        }

    region_terms = ("mainland", "island")

    options = []
    for label, rows in results_by_option.items():
        area_was_stated = any(
            opt.area is not None
            for opt in (entities.comparison_options or [])
            if opt.label == label
        )
        implies_region = any(term in label.lower() for term in region_terms)
        caveat = None
        if implies_region and not area_was_stated:
            caveat = "Region-level filtering isn't available yet — showing citywide matches for the other criteria, not limited to this region."

        options.append({
            "label": label,
            "matched_count": len(rows),
            "caveat": caveat,
            "cheapest": format_price(rows[0]["price_ngn"], rows[0]["price_period"]) if rows else None,
            "results": [
                {
                    "title": r["title"],
                    "price": format_price(r["price_ngn"], r["price_period"]),
                    "area": r["area"],
                    "bedrooms": r["bedrooms"],
                    "url": r["url"],
                }
                for r in rows
            ],
        })

    populated = [o for o in options if o["matched_count"] > 0]
    if populated:
        cheapest_label = min(
            populated,
            key=lambda o: results_by_option[o["label"]][0]["price_ngn"],
        )["label"]
        message = f"'{cheapest_label}' has the most affordable live match of the options compared."
    else:
        message = "No live listings matched any of the options compared."

    return {
        "query_type": entities.query_type.value,
        "message": message,
        "options": options,
    }