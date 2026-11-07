import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from advice.advice_engine import answer_advice_query
from config import REQUEST_TIMEOUT_SECONDS
from conversation.session import load_session_entities, save_session_entities
from observability.tracer import start_trace, trace_stage
from query.answer import ChatAnswer, Clarification, compute_contract
from query.entities import EntityExtractionError, QueryType, extract_entities
from query.query_engine import fetch_rent_stats, run_affordability, run_comparison, run_lookup
from query.response import build_affordability_response, build_comparison_response, build_lookup_response

logger.add("logs/app.log", rotation="10 MB", retention="14 days", level="INFO")

app = FastAPI(title="Lagos House Hunter AI")

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


def _error_payload(message: str, request_id: str, latency_ms: float, status: str = "error") -> dict:
    ans = ChatAnswer(status=status, request_id=request_id, latency_ms=round(latency_ms, 2), message=message)
    return {"message": message, "answer": ans.model_dump(mode="json")}


@app.post("/query")
def query_listings(request: QueryRequest, http_request: Request):
    t0 = time.perf_counter()
    request_id = request.request_id or str(uuid.uuid4())
    key = _resolve_key(request, http_request.headers)
    if key and key in _idempotency_cache:
        return _idempotency_cache[key]
    start_trace()
    previous_entities = load_session_entities(request.session_id) if request.session_id else None
    try:
        with trace_stage("extraction"):
            entities = extract_entities(request.query, previous_entities=previous_entities)
    except EntityExtractionError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("Entity extraction failed for query='{}': {}", request.query, exc)
        payload = _error_payload("Could not understand the query.", request_id, latency_ms, "error")
        if key:
            _idempotency_cache[key] = payload
        return payload
    if not request.query or not str(request.query).strip():
        latency_ms = (time.perf_counter() - t0) * 1000
        return _error_payload("Query text is empty.", request_id, latency_ms, "clarification")
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
        else:
            latency_ms = (time.perf_counter() - t0) * 1000
            return _error_payload(f"Query type '{entities.query_type.value}' is not supported.", request_id, latency_ms, "out_of_scope")
    except Exception as exc:
        logger.error("Query failed for query='{}': {}", request.query, exc)
        latency_ms = (time.perf_counter() - t0) * 1000
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
    return await loop.run_in_executor(None, lambda: query_listings(request, http_request))


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
