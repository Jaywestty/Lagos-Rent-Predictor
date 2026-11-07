from datetime import UTC, datetime

from pydantic import BaseModel, Field

from config import LIMITED_CONFIDENCE_MAX, LOW_DATA_MAX, PERCENTILE_MIN_COUNT


class RangeStat(BaseModel):
    min: float | None = None
    max: float | None = None
    median: float | None = None
    count: int = 0


class Breakdown(BaseModel):
    self_contain: RangeStat | None = None
    one_bedroom: RangeStat | None = None
    two_bedroom: RangeStat | None = None
    three_bedroom: RangeStat | None = None


class Headline(BaseModel):
    min: float | None = None
    max: float | None = None


class Clarification(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)


class ChatAnswer(BaseModel):
    version: int = 1
    status: str = "error"
    location: str | None = None
    area: str | None = None
    property_type: str | None = None
    currency: str = "NGN"
    rent_period: str = "yearly"
    headline: Headline | None = None
    breakdown: Breakdown | None = None
    median: float | None = None
    listing_count: int = 0
    range_method: str = "min_max"
    data_as_of: str | None = None
    last_updated_days: int | None = None
    confidence: str = "insufficient"
    clarification: Clarification | None = None
    follow_ups: list[str] = Field(default_factory=list)
    request_id: str = ""
    latency_ms: float = 0.0
    message: str = ""
    comparison: dict | None = None


def compute_contract(values_by_bucket, all_values, area=None, location=None, property_type=None, data_as_of=None, request_id="", latency_ms=0.0):
    from query.stats import classify, summarize_values

    total = len([v for v in all_values if v is not None])
    status, confidence = classify(total, LOW_DATA_MAX, LIMITED_CONFIDENCE_MAX)
    summary = summarize_values(all_values, PERCENTILE_MIN_COUNT) if all_values else None
    headline = None
    median_val = None
    method = "min_max"
    if summary:
        headline = Headline(min=summary["min"], max=summary["max"])
        median_val = summary["median"]
        method = summary["range_method"]
    breakdown = Breakdown()
    if values_by_bucket:
        map_key = {
            "self_contain": "self_contain",
            "one_bedroom": "one_bedroom",
            "two_bedroom": "two_bedroom",
            "three_bedroom": "three_bedroom",
        }
        for bucket, vals in values_by_bucket.items():
            s = summarize_values(vals, PERCENTILE_MIN_COUNT) if vals else None
            if s:
                setattr(breakdown, map_key[bucket], RangeStat(min=s["min"], max=s["max"], median=s["median"], count=s["count"]))
    last_days = None
    data_as_of_str = None
    if data_as_of:
        try:
            dt = data_as_of if isinstance(data_as_of, datetime) else datetime.fromisoformat(str(data_as_of).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            data_as_of_str = dt.isoformat()
            last_days = int((datetime.now(UTC) - dt).total_seconds() // 86400)
        except Exception:
            pass
    return ChatAnswer(
        status=status,
        location=location or area,
        area=area,
        property_type=property_type,
        headline=headline,
        breakdown=breakdown,
        median=median_val,
        listing_count=total,
        range_method=method,
        data_as_of=data_as_of_str,
        last_updated_days=last_days,
        confidence=confidence,
        follow_ups=_follow_ups(area, property_type, status),
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
        message="",
    )


def _follow_ups(area, property_type, status):
    ups = []
    if area:
        ups.append(f"What about a 2 bedroom in {area}?")
        ups.append(f"How much is rent in {area} for a 3 bedroom?")
    else:
        ups.append("How much is rent in Gbagada?")
        ups.append("How much is rent in Yaba?")
    if property_type:
        ups.append("What other property types are available?")
    return ups[:3]
