import asyncio
import contextvars
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from advice.advice_engine import answer_advice_query
from config import REQUEST_TIMEOUT_SECONDS
from conversation.session import load_session_entities, save_session_entities
from observability.events import install_http_middleware, record_error
from observability.tracer import start_trace, trace_stage
from query.answer import ChatAnswer, Clarification, compute_contract
from query.entities import EntityExtractionError, QueryType, extract_entities
from query.query_engine import fetch_rent_stats, run_affordability, run_comparison, run_lookup
from query.response import build_affordability_response, build_comparison_response, build_lookup_response

logger.add("logs/app.log", rotation="10 MB", retention="14 days", level="INFO")



@asynccontextmanager
async def _lifespan(_app):
    # Close scrape runs whose heartbeat has gone silent (for example killed by a previous restart),
    # so admin never shows a ghost run. Heartbeat-based, so it is safe with several servers.
    try:
        from scraping.runner import mark_interrupted_runs

        await asyncio.get_running_loop().run_in_executor(None, mark_interrupted_runs)
    except Exception as exc:
        logger.warning("could not close orphaned scrape runs: {}", exc)
    yield


app = FastAPI(title="Adugbo AI", lifespan=_lifespan)
install_http_middleware(app)

# Optional: only needed when a browser on another origin calls this API directly (for example
# VITE_API_BASE_URL pointing at Render). The recommended Vercel setup proxies /api instead.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=True,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"],
                       allow_headers=["Content-Type", "X-Client-Id", "X-Request-Key", "X-CSRF-Token", "X-Request-Id"])

from entrypoints.public_api import router as public_router  # noqa: E402

app.include_router(public_router)

try:
    from admin.api import router as admin_router

    app.include_router(admin_router)
    from admin.observability import router as admin_obs_router

    app.include_router(admin_obs_router)
except Exception as _adm_exc:
    logger.warning("admin router not loaded: {}", _adm_exc)

_idempotency_cache: dict[str, dict] = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    request_key: str | None = None
    request_id: str | None = None


def _resolve_key(request: QueryRequest, headers):
    if request.request_key:
        return request.request_key
    return headers.get("x-request-key") or headers.get("Idempotency-Key")


def _enrich_with_contract(legacy: dict, entities, request_id: str, latency_ms: float) -> dict:
    try:
        stats = fetch_rent_stats(area=entities.area, beds=entities.beds, property_type=entities.property_type)
    except Exception as exc:
        logger.warning("stats fetch failed: {}", exc)
        stats = {"buckets": {}, "all_values": [], "data_as_of": None, "area": entities.area}
    contract = compute_contract(
        stats["buckets"],
        stats["all_values"],
        area=stats.get("area") or entities.area,
        property_type=entities.property_type,
        data_as_of=stats.get("data_as_of"),
        request_id=request_id,
        latency_ms=latency_ms,
    )
    payload = legacy if isinstance(legacy, dict) else {"message": str(legacy)}
    payload["answer"] = contract.model_dump(mode="json")
    payload["answer"]["message"] = payload.get("message", "")
    if entities.query_type == QueryType.COMPARISON:
        comparison = _comparison_contract(entities)
        payload["answer"]["comparison"] = comparison
        counts = [o["count"] for o in (comparison or {}).get("options", [])]
        payload["answer"]["status"] = "success" if any(counts) else "no_data"
        payload["answer"]["listing_count"] = sum(counts)
    if entities.query_type == QueryType.AFFORDABILITY:
        payload["answer"]["budget"] = _budget_contract(payload, entities)
    if payload["answer"].get("status") == "no_data":
        payload["answer"]["no_data"] = _no_data_help(entities)
    is_sale = bool(entities.listing_type and entities.listing_type.value == "sale")
    if is_sale:
        payload["answer"]["status"] = "out_of_scope"
        payload["answer"]["confidence"] = "insufficient"
    if not is_sale and not entities.area and not entities.beds and not entities.property_type and not entities.budget_ngn and entities.query_type in (QueryType.LOOKUP,):
        overall = compute_contract({}, [], request_id=request_id, latency_ms=latency_ms)
        if overall.listing_count == 0:
            payload["answer"]["status"] = "clarification"
            payload["answer"]["clarification"] = Clarification(
                question="Which area are you looking for?", options=["Gbagada", "Yaba", "Ikeja", "Surulere"]
            ).model_dump(mode="json")
    return payload


MAX_QUESTION_CHARS = 400
OUT_OF_SCOPE_MESSAGE = (
    "I'm Adugbo AI, and I can only help with renting in Lagos: prices by area and size, budgets, comparing areas, "
    "and practical renting advice. Try something like \"2 bedroom rent in Yaba\"."
)
# Obvious attempts to override the assistant. Caught before any AI call, so they cost nothing.
_INJECTION = re.compile(
    r"((ignore|disregard|forget|override) (?:(?:all|any|the|your|my|of|these|those|previous|prior|above|earlier)\s+){0,4}"
    r"(instructions|prompts?|rules|guidelines)"
    r"|(reveal|show|print|repeat|leak) (me )?(your|the) (system )?(prompt|instructions|rules)"
    r"|system prompt|developer mode|jailbreak|you are now|pretend (to be|you are)|act as (an?|my) )",
    re.IGNORECASE,
)


def _out_of_scope_payload(request_id: str, latency_ms: float, message: str = OUT_OF_SCOPE_MESSAGE) -> dict:
    payload = _error_payload(message, request_id, latency_ms, "out_of_scope")
    payload["query_type"] = "out_of_scope"
    return payload


def _comparison_contract(entities) -> dict | None:
    """Side-by-side figures for each option in a comparison, all from listings."""
    from query.stats import classify, summarize_values

    options = []
    for opt in entities.comparison_options or []:
        try:
            stats = fetch_rent_stats(area=opt.area, beds=opt.beds, property_type=opt.property_type)
        except Exception as exc:
            logger.warning("comparison stats failed for {}: {}", opt.label, exc)
            continue
        summary = summarize_values(stats["all_values"]) if stats["all_values"] else None
        _, confidence = classify(len(stats["all_values"]))
        region_only = opt.area is None and any(t in opt.label.lower() for t in ("mainland", "island"))
        options.append({
            "label": opt.label,
            "area": stats.get("area") or opt.area,
            "beds": opt.beds,
            "property_type": opt.property_type,
            "median": summary["median"] if summary else None,
            "min": summary["min"] if summary else None,
            "max": summary["max"] if summary else None,
            "count": summary["count"] if summary else 0,
            "confidence": confidence,
            "caveat": "Area-level only: there is no mainland/island filter, so this covers all of Lagos for this size." if region_only else None,
        })
    return {"options": options} if options else None


def _no_data_help(entities) -> dict:
    """Why a search found nothing, and what to try instead: a spelling fix, other sizes in the
    same area, or the busiest covered areas. Everything here comes from the listings table."""
    from rapidfuzz import fuzz, process

    from query.market import list_areas

    asked = (entities.area or "").strip()
    try:
        covered = list_areas()
    except Exception as exc:
        logger.warning("no-data help unavailable: {}", exc)
        return {"area": asked or None}
    names = [a["area"] for a in covered]
    by_name = {a["area"].lower(): a for a in covered}
    help_: dict = {"area": asked or None, "covered": False, "area_listings": 0, "did_you_mean": None,
                   "suggestions": names[:6]}
    if asked:
        exact = by_name.get(asked.lower())
        if exact:
            help_["covered"] = True
            help_["area"] = exact["area"]
            help_["area_listings"] = exact["listing_count"]
        else:
            match = process.extractOne(asked, names, scorer=fuzz.WRatio, score_cutoff=86)
            if match and match[0].lower() != asked.lower():
                help_["did_you_mean"] = match[0]
    return help_


def _budget_contract(legacy: dict, entities) -> dict | None:
    from query.query_engine import annual_budget

    yearly = annual_budget(entities)
    if yearly is None:
        return None
    period = getattr(entities.budget_period, "value", entities.budget_period)
    areas = [{"area": a.get("area"), "count": a.get("matched_count", 0)} for a in legacy.get("areas") or []]
    stretch = sum(len(s.get("results") or []) for s in legacy.get("stretch_options") or [])
    return {
        "amount": float(entities.budget_ngn),
        "period": period,
        "yearly": yearly,
        "within_count": int(legacy.get("matched_count") or 0),
        "areas": areas[:6],
        "stretch_count": stretch,
    }


_PRICE_QUESTION = re.compile(r"\b(how much|price|prices|cost|rent|rents|budget|afford|cheap|expensive)\b", re.IGNORECASE)
_BEDS_IN_TEXT = re.compile(r"\b([1-9])\s*-?\s*(?:bed|bedroom|bedrooms|br)\b", re.IGNORECASE)


def _area_named_in(question: str) -> str | None:
    """A known Lagos area mentioned in the text, longest names first ("Victoria Island" before "Island")."""
    from scraping.location_normalizer import KNOWN_AREAS

    text = f" {(question or '').lower()} "
    for area in sorted(KNOWN_AREAS, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(area.lower())}(?![a-z])", text):
            return area
    return None


def _route_price_questions_to_listings(question: str, entities):
    """A price question about a known area must be answered from listings, never from web search.

    The extractor sometimes labels "how much is a 2 bedroom in Lekki?" as advice; web results then
    quote sale prices or stale figures. Rent figures must stay traceable to listings.
    """
    if entities.query_type != QueryType.ADVICE or not _PRICE_QUESTION.search(question or ""):
        return entities
    area = entities.area or _area_named_in(question)
    if not area:
        return entities
    update: dict = {"query_type": QueryType.LOOKUP, "area": area}
    if entities.beds is None:
        m = _BEDS_IN_TEXT.search(question)
        if m:
            update["beds"] = int(m.group(1))
    logger.info("Re-routing advice-classified price question to lookup: {}", question)
    return entities.model_copy(update=update)


def _error_payload(message: str, request_id: str, latency_ms: float, status: str = "error") -> dict:
    ans = ChatAnswer(status=status, request_id=request_id, latency_ms=round(latency_ms, 2), message=message)
    return {"message": message, "answer": ans.model_dump(mode="json")}


@app.post("/query")
def query_listings(request: QueryRequest, http_request: Request):
    t0 = time.perf_counter()
    from observability.events import current_request_id

    request_id = request.request_id or current_request_id.get() or str(uuid.uuid4())
    key = _resolve_key(request, http_request.headers)
    if key and key in _idempotency_cache:
        return _idempotency_cache[key]
    question = (request.query or "").strip()
    if not question:
        return _error_payload("Type a question about rent in Lagos.", request_id, (time.perf_counter() - t0) * 1000, "clarification")
    if len(question) > MAX_QUESTION_CHARS:
        return _error_payload(f"Please keep your question under {MAX_QUESTION_CHARS} characters.", request_id,
                              (time.perf_counter() - t0) * 1000, "clarification")
    if _INJECTION.search(question):
        logger.info("Blocked instruction-override attempt: {}", question[:120])
        return _out_of_scope_payload(request_id, (time.perf_counter() - t0) * 1000)
    start_trace()
    previous_entities = load_session_entities(request.session_id) if request.session_id else None
    try:
        with trace_stage("extraction"):
            entities = extract_entities(request.query, previous_entities=previous_entities)
    except EntityExtractionError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("Entity extraction failed for query='{}': {}", request.query, exc)
        record_error("extraction", exc, route="/query")
        payload = _error_payload("Could not understand the query.", request_id, latency_ms, "error")
        if key:
            _idempotency_cache[key] = payload
        return payload
    except Exception as exc:
        # The AI service is unreachable or failed (network, outage, rate limit). Not cached, so a retry works.
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("Entity extraction service failed for query='{}': {}", request.query, exc)
        return _error_payload("We couldn't reach the service that reads your question. Please try again in a moment.",
                              request_id, latency_ms, "error")
    if not request.query or not str(request.query).strip():
        latency_ms = (time.perf_counter() - t0) * 1000
        return _error_payload("Query text is empty.", request_id, latency_ms, "clarification")
    if entities.query_type == QueryType.OUT_OF_SCOPE:
        return _out_of_scope_payload(request_id, (time.perf_counter() - t0) * 1000)
    entities = _route_price_questions_to_listings(request.query, entities)
    query_type_value = entities.query_type.value
    try:
        if entities.query_type == QueryType.LOOKUP:
            with trace_stage("query_execution", query_type_value):
                listings = run_lookup(entities)
            result = build_lookup_response(entities, listings)
        elif entities.query_type == QueryType.AFFORDABILITY:
            with trace_stage("query_execution", query_type_value):
                aff_result = run_affordability(entities)
            result = build_affordability_response(entities, aff_result)
        elif entities.query_type == QueryType.COMPARISON:
            with trace_stage("query_execution", query_type_value):
                comp_result = run_comparison(entities)
            result = build_comparison_response(entities, comp_result)
        elif entities.query_type == QueryType.ADVICE:
            with trace_stage("advice_synthesis", query_type_value):
                result = answer_advice_query(entities.resolved_query or request.query)
            if isinstance(result, dict) and result.get("query_type") == "out_of_scope":
                return _out_of_scope_payload(request_id, (time.perf_counter() - t0) * 1000)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000
            return _error_payload(f"Query type '{entities.query_type.value}' is not supported.", request_id, latency_ms, "out_of_scope")
    except Exception as exc:
        logger.error("Query failed for query='{}': {}", request.query, exc)
        record_error("query", exc, route="/query")
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            from analytics_logging import log_question

            log_question(request_id, session_id=request.session_id, raw_question=request.query, status="error", listing_count=0, confidence="insufficient", latency_ms=latency_ms, error_code="query_failed")
        except Exception as log_exc:
            logger.warning("failure log failed: {}", log_exc)
        return _error_payload("Search failed. Try again.", request_id, latency_ms, "error")
    latency_ms = (time.perf_counter() - t0) * 1000
    if request.session_id:
        save_session_entities(request.session_id, entities)
    if entities.query_type == QueryType.ADVICE:
        payload = result if isinstance(result, dict) else {"message": str(result)}
        payload["answer"] = ChatAnswer(
            status="success", location=entities.area, area=entities.area, property_type=entities.property_type,
            request_id=request_id, latency_ms=round(latency_ms, 2), message=payload.get("answer", ""),
        ).model_dump(mode="json")
    else:
        payload = _enrich_with_contract(result, entities, request_id, latency_ms)
    try:
        from analytics_logging import log_question

        ans = payload.get("answer", {}) if isinstance(payload, dict) else {}
        log_question(
            request_id,
            session_id=request.session_id,
            raw_question=request.query,
            parsed_location=entities.area,
            parsed_property_type=entities.property_type,
            status=ans.get("status", "error"),
            listing_count=ans.get("listing_count", 0),
            confidence=ans.get("confidence"),
            latency_ms=latency_ms,
            error_code=None if ans.get("status") not in ("error",) else "query_failed",
        )
    except Exception as exc:
        logger.warning("question log failed: {}", exc)
    try:
        # Chat history: the session id doubles as the thread id; the device id owns the thread.
        from conversation.history import save_exchange

        save_exchange(http_request.headers.get("x-client-id"), request.session_id, request.query, payload)
    except Exception as exc:
        logger.warning("chat history save failed: {}", exc)
    if key:
        _idempotency_cache[key] = payload
    return payload


@app.post("/api/chat")
async def api_chat(request: QueryRequest, http_request: Request):
    try:
        return await asyncio.wait_for(_run_chat_async(request, http_request), timeout=REQUEST_TIMEOUT_SECONDS)
    except TimeoutError:
        return _error_payload("Request timed out. Try again.", request.request_id or str(uuid.uuid4()), 0.0, "error")


async def _run_chat_async(request: QueryRequest, http_request: Request):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()  # keep the request id for observability inside the worker thread
    return await loop.run_in_executor(None, lambda: ctx.run(query_listings, request, http_request))


@app.post("/api/chat/stream")
async def api_chat_stream(request: QueryRequest, http_request: Request):
    async def gen():
        rid = request.request_id or str(uuid.uuid4())
        yield f"event: status\ndata: {json.dumps({'type': 'status', 'request_id': rid, 'text': 'Understanding your question...'})}\n\n"
        try:
            result = await asyncio.wait_for(_run_chat_async(request, http_request), timeout=REQUEST_TIMEOUT_SECONDS)
        except TimeoutError:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'request_id': rid, 'message': 'Request timed out.'})}\n\n"
            return
        yield f"event: delta\ndata: {json.dumps({'type': 'delta', 'request_id': rid, 'text': 'Checking live listings...'})}\n\n"
        yield f"event: final\ndata: {json.dumps({'type': 'final', 'request_id': rid, 'result': result})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
