"""Area guides: what it is like to live in a Lagos area.

Qualitative sections (commute, flooding, power, security...) are built only from
web search results (Tavily) and summarised by Groq with source citations. They
are cached in area_guides for GUIDE_TTL_DAYS. Rent numbers never come from the
LLM: the API attaches them from listings (query.market.area_stats).
"""

import json
from datetime import UTC, datetime, timedelta

from groq import Groq
from loguru import logger
from sqlalchemy import text

from advice.search_client import SearchError, search_advice
from config import GROQ_API_KEY, GROQ_MODEL
from db import get_engine
from observability.events import groq_chat, record_error

GUIDE_TTL_DAYS = 30
SECTIONS = [
    ("overview", "Overview"),
    ("commute", "Commute and traffic"),
    ("flooding", "Flooding and drainage"),
    ("power", "Electricity and water"),
    ("security", "Safety and security"),
    ("amenities", "Markets, schools and amenities"),
    ("lifestyle", "Who it suits"),
]
SEARCHES = [
    "{area} Lagos neighbourhood guide living in {area} cost of living amenities",
    "{area} Lagos traffic commute flooding electricity security residents",
]

GUIDE_PROMPT = """You write short, practical neighbourhood guides for people looking to rent in {area}, Lagos, Nigeria.

Use ONLY the numbered search results provided. Never invent facts, prices, distances or names.
Do not state rent prices at all; rent figures come from a separate listings database.
If the sources say nothing useful about a section, set its text to null.

Return a JSON object with exactly these keys:
{keys}
Each key maps to an object: {{"text": "<2 to 4 plain sentences or null>", "sources": [<source numbers used>]}}.
Also include "watch_outs": a list of up to 4 short strings (each one sentence, grounded in the sources),
and "watch_out_sources": a list of source numbers.
"""


class GuideUnavailable(Exception):
    pass


def _cached(area: str) -> dict | None:
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT content, sources, model, generated_at FROM area_guides WHERE lower(area) = lower(:a)"),
                           {"a": area}).mappings().first()
    if not row:
        return None
    return {"content": row["content"], "sources": row["sources"], "model": row["model"],
            "generated_at": row["generated_at"]}


def _store(area: str, content: dict, sources: list[dict]) -> None:
    with get_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO area_guides (area, content, sources, model, generated_at)
            VALUES (:a, CAST(:c AS JSONB), CAST(:s AS JSONB), :m, now())
            ON CONFLICT (area) DO UPDATE SET content = EXCLUDED.content, sources = EXCLUDED.sources,
              model = EXCLUDED.model, generated_at = now()
        """), {"a": area, "c": json.dumps(content), "s": json.dumps(sources), "m": GROQ_MODEL})


def _gather_sources(area: str) -> list[dict]:
    seen, results = set(), []
    for template in SEARCHES:
        try:
            for r in search_advice(template.format(area=area), purpose="guide", max_results=6):
                if r["url"] in seen:
                    continue
                seen.add(r["url"])
                results.append(r)
        except SearchError as exc:
            logger.warning("guide search failed for {}: {}", area, exc)
    return results[:10]


def _clean(raw: dict, n_sources: int) -> dict:
    def ids(v):
        return sorted({int(i) for i in (v or []) if str(i).isdigit() and 1 <= int(i) <= n_sources})

    sections = []
    for key, title in SECTIONS:
        item = raw.get(key) or {}
        body = item.get("text") if isinstance(item, dict) else None
        sections.append({"key": key, "title": title, "text": body.strip() if isinstance(body, str) and body.strip() else None,
                         "sources": ids(item.get("sources") if isinstance(item, dict) else [])})
    watch = [w.strip() for w in (raw.get("watch_outs") or []) if isinstance(w, str) and w.strip()][:4]
    return {"sections": sections, "watch_outs": watch, "watch_out_sources": ids(raw.get("watch_out_sources"))}


def generate_guide(area: str) -> dict:
    if not GROQ_API_KEY:
        raise GuideUnavailable("GROQ_API_KEY is not set")
    results = _gather_sources(area)
    if not results:
        raise GuideUnavailable("No web sources found for this area")
    context = "\n\n".join(f"[{i}] {r['title']}\nURL: {r['url']}\n{(r['content'] or '')[:1500]}" for i, r in enumerate(results, 1))
    keys = ", ".join(k for k, _ in SECTIONS)
    try:
        resp = groq_chat(
            Groq(api_key=GROQ_API_KEY), "guide",
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": GUIDE_PROMPT.format(area=area, keys=keys)},
                      {"role": "user", "content": f"Search results for {area}:\n\n{context}"}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        record_error("guide", exc)
        raise GuideUnavailable("Could not write the guide right now") from exc
    content = _clean(raw, len(results))
    sources = [{"n": i, "title": r["title"], "url": r["url"]} for i, r in enumerate(results, 1)]
    _store(area, content, sources)
    return {"content": content, "sources": sources, "model": GROQ_MODEL, "generated_at": datetime.now(UTC)}


def get_guide(area: str, refresh: bool = False) -> dict:
    """Cached guide if fresh, otherwise generate. Falls back to a stale cache when generation fails."""
    cached = _cached(area)
    fresh = cached and cached["generated_at"] and cached["generated_at"] > datetime.now(UTC) - timedelta(days=GUIDE_TTL_DAYS)
    if cached and fresh and not refresh:
        return {**cached, "cached": True}
    try:
        return {**generate_guide(area), "cached": False}
    except GuideUnavailable:
        if cached:
            return {**cached, "cached": True, "stale": True}
        raise
