"""Public read APIs for the app's pages, plus per-device saved searches and favourites.

Saved items are keyed by an anonymous client id (a random UUID the browser keeps
and sends as X-Client-Id). No account, no personal data.
"""

import re
import uuid
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from db import get_engine
from observability.events import record_feature
from query import market
from query.response import listing_card

router = APIRouter(prefix="/api", tags=["public"])

ALLOWED_EVENTS = {"move_in_calculator", "compare_view", "guide_view", "trend_view", "listing_click", "chip_click", "answer_feedback"}


def _client(x_client_id: str | None) -> str:
    try:
        return str(uuid.UUID(x_client_id or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Client-Id header must be a UUID") from exc


@router.get("/areas")
def areas(min_count: int = Query(1, ge=1, le=50)):
    return {"items": market.list_areas(min_count=min_count)}


@router.get("/areas/{area}")
def area_detail(area: str, x_client_id: str | None = Header(default=None)):
    stats = market.area_stats(area)
    if stats["listing_count"] == 0:
        raise HTTPException(status_code=404, detail="No listings for this area yet")
    return {**stats, "trend": market.trends(stats["area"], "all")}


@router.get("/areas/{area}/trends")
def area_trends(area: str, bucket: str = "all", weeks: int = Query(26, ge=1, le=104)):
    return market.trends(area, bucket, weeks)


@router.get("/areas/{area}/listings")
def area_listings(area: str, beds: int | None = Query(None, ge=0, le=10), limit: int = Query(12, ge=1, le=50)):
    filters = [market._BASE_WHERE, "lower(area) = lower(:area)"]
    params: dict[str, object] = {"area": market.canonical_area(area), "limit": limit}
    if beds is not None:
        filters.append("bedrooms = :beds")
        params["beds"] = beds
    sql = f"""
        SELECT id, source, url, title, price_ngn, price_period, area, subarea, bedrooms, bathrooms, property_type, listed_at,
               (CASE WHEN price_period = 'month' THEN price_ngn * 12 ELSE price_ngn END) AS annual_price
        FROM listings WHERE {' AND '.join(filters)}
        ORDER BY listed_at DESC NULLS LAST, id DESC LIMIT :limit
    """
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"items": [listing_card(dict(r)) for r in rows]}


@router.get("/areas/{area}/guide")
def area_guide(area: str, refresh: bool = False, x_client_id: str | None = Header(default=None)):
    from advice.area_guide import GuideUnavailable, get_guide

    stats = market.area_stats(area)
    name = stats["area"] or market.canonical_area(area) or area
    record_feature("guide_view", x_client_id, area=name)
    try:
        guide = get_guide(name, refresh=refresh)
    except GuideUnavailable as exc:
        return {"area": name, "stats": stats, "guide": None, "guide_error": str(exc)}
    generated = guide.get("generated_at")
    return {
        "area": name,
        "stats": stats,
        "guide": guide["content"],
        "sources": guide["sources"],
        "generated_at": generated.isoformat() if isinstance(generated, datetime) else generated,
        "stale": bool(guide.get("stale")),
    }


@router.get("/compare")
def compare(areas: str = Query(..., description="Comma separated, up to 3"), beds: int | None = Query(None, ge=1, le=3),
            x_client_id: str | None = Header(default=None)):
    names = [a.strip() for a in areas.split(",") if a.strip()][:3]
    if len(names) < 2:
        raise HTTPException(status_code=400, detail="Pick at least two areas")
    record_feature("compare_view", x_client_id, areas=names, beds=beds)
    return market.compare(names, beds)


class SavedSearchIn(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    query: str | None = Field(default=None, max_length=500)
    area: str | None = Field(default=None, max_length=80)
    bedrooms: int | None = Field(default=None, ge=0, le=10)
    max_budget_ngn: float | None = Field(default=None, ge=0)


@router.get("/me/saved-searches")
def list_saved(x_client_id: str | None = Header(default=None)):
    cid = _client(x_client_id)
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT id, label, query, area, bedrooms, max_budget_ngn, created_at FROM saved_searches WHERE client_id = :c ORDER BY created_at DESC"),
                            {"c": cid}).mappings().all()
    items = []
    for r in rows:
        item = dict(r)
        if item["area"]:
            s = market.area_stats(item["area"])
            focus = s["breakdown"].get(market.BEDS_TO_BUCKET.get(item["bedrooms"])) if item["bedrooms"] in market.BEDS_TO_BUCKET else s["overall"]
            item["current"] = {"listing_count": s["listing_count"], "median_ngn": focus["median"] if focus else None, "data_as_of": s["data_as_of"]}
        items.append(item)
    return {"items": items}


@router.post("/me/saved-searches", status_code=201)
def create_saved(body: SavedSearchIn, x_client_id: str | None = Header(default=None)):
    cid = _client(x_client_id)
    with get_engine().begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM saved_searches WHERE client_id = :c"), {"c": cid}).scalar() or 0
        if count >= 50:
            raise HTTPException(status_code=409, detail="You can keep up to 50 saved searches")
        row = conn.execute(text("""INSERT INTO saved_searches (client_id, label, query, area, bedrooms, max_budget_ngn)
                                   VALUES (:c, :label, :query, :area, :bedrooms, :max_budget_ngn) RETURNING id"""),
                           {"c": cid, **body.model_dump()}).scalar()
    record_feature("saved_search_create", cid, area=body.area)
    return {"id": row}


@router.delete("/me/saved-searches/{item_id}")
def delete_saved(item_id: int, x_client_id: str | None = Header(default=None)):
    cid = _client(x_client_id)
    with get_engine().begin() as conn:
        n = conn.execute(text("DELETE FROM saved_searches WHERE id = :id AND client_id = :c"), {"id": item_id, "c": cid}).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


class FavoriteIn(BaseModel):
    url: str = Field(min_length=8, max_length=1000)
    title: str | None = Field(default=None, max_length=300)
    price_text: str | None = Field(default=None, max_length=80)
    annual_price_ngn: float | None = Field(default=None, ge=0)
    area: str | None = Field(default=None, max_length=80)
    bedrooms: int | None = Field(default=None, ge=0, le=20)
    source: str | None = Field(default=None, max_length=40)


@router.get("/me/favorites")
def list_favorites(x_client_id: str | None = Header(default=None)):
    cid = _client(x_client_id)
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT f.id, f.url, f.title, f.price_text, f.annual_price_ngn, f.area, f.bedrooms, f.source, f.created_at,
                   (l.id IS NOT NULL) AS still_listed, l.listed_at
            FROM favorites f LEFT JOIN listings l ON l.url = f.url
            WHERE f.client_id = :c ORDER BY f.created_at DESC
        """), {"c": cid}).mappings().all()
    return {"items": [dict(r) for r in rows]}


@router.post("/me/favorites", status_code=201)
def add_favorite(body: FavoriteIn, x_client_id: str | None = Header(default=None)):
    cid = _client(x_client_id)
    if not re.match(r"^https://(nigeriapropertycentre\.com|propertypro\.ng)/", body.url):
        raise HTTPException(status_code=400, detail="Only listings from supported sites can be saved")
    with get_engine().begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM favorites WHERE client_id = :c"), {"c": cid}).scalar() or 0
        if count >= 200:
            raise HTTPException(status_code=409, detail="You can keep up to 200 favourites")
        row = conn.execute(text("""
            INSERT INTO favorites (client_id, url, title, price_text, annual_price_ngn, area, bedrooms, source)
            VALUES (:c, :url, :title, :price_text, :annual_price_ngn, :area, :bedrooms, :source)
            ON CONFLICT (client_id, url) DO UPDATE SET title = EXCLUDED.title RETURNING id
        """), {"c": cid, **body.model_dump()}).scalar()
    record_feature("favorite_add", cid, area=body.area)
    return {"id": row}


@router.delete("/me/favorites/{item_id}")
def delete_favorite(item_id: int, x_client_id: str | None = Header(default=None)):
    cid = _client(x_client_id)
    with get_engine().begin() as conn:
        n = conn.execute(text("DELETE FROM favorites WHERE id = :id AND client_id = :c"), {"id": item_id, "c": cid}).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


class EventIn(BaseModel):
    feature: str
    detail: dict = Field(default_factory=dict)


@router.post("/events", status_code=202)
def track(body: EventIn, x_client_id: str | None = Header(default=None)):
    if body.feature not in ALLOWED_EVENTS:
        raise HTTPException(status_code=400, detail="Unknown event")
    detail = {k: v for k, v in list(body.detail.items())[:10] if isinstance(v, str | int | float | bool | None)}
    record_feature(body.feature, x_client_id if x_client_id and len(x_client_id) <= 64 else None, **detail)
    return {"ok": True}


# Chat history (per device, see conversation/history.py)


@router.get("/me/threads")
def list_chat_threads(x_client_id: str | None = Header(default=None)):
    from conversation.history import list_threads

    return {"items": list_threads(_client(x_client_id))}


@router.get("/me/threads/{thread_id}")
def get_chat_thread(thread_id: str, x_client_id: str | None = Header(default=None)):
    from conversation.history import get_thread, valid_uuid

    tid = valid_uuid(thread_id)
    thread = get_thread(_client(x_client_id), tid) if tid else None
    if not thread:
        raise HTTPException(status_code=404, detail="Chat not found")
    return thread


@router.delete("/me/threads/{thread_id}")
def delete_chat_thread(thread_id: str, x_client_id: str | None = Header(default=None)):
    from conversation.history import delete_thread, valid_uuid

    tid = valid_uuid(thread_id)
    if not tid or not delete_thread(_client(x_client_id), tid):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"ok": True}


@router.delete("/me/threads")
def delete_all_chat_threads(x_client_id: str | None = Header(default=None)):
    from conversation.history import delete_all_threads

    return {"ok": True, "deleted": delete_all_threads(_client(x_client_id))}
