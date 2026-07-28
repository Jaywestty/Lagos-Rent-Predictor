from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from typing import Optional

from entities import EntityExtractionError, QueryType, extract_entities
from query_engine import run_lookup, run_affordability, run_comparison
from response import build_lookup_response, build_affordability_response, build_comparison_response
from session import load_session_entities, save_session_entities

logger.add("logs/app.log", rotation="10 MB", retention="14 days", level="INFO")

app = FastAPI(title="Lagos House Hunter AI")


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


@app.post("/query")
def query_listings(request: QueryRequest):
    previous_entities = load_session_entities(request.session_id) if request.session_id else None

    try:
        entities = extract_entities(request.query, previous_entities=previous_entities)
    except EntityExtractionError as exc:
        logger.error("Entity extraction failed for query='{}': {}", request.query, exc)
        raise HTTPException(status_code=422, detail="Could not understand the query.") from exc

    try:
        if entities.query_type == QueryType.LOOKUP:
            listings = run_lookup(entities)
            result = build_lookup_response(entities, listings)
        elif entities.query_type == QueryType.AFFORDABILITY:
            aff_result = run_affordability(entities)
            result = build_affordability_response(entities, aff_result)
        elif entities.query_type == QueryType.COMPARISON:
            comp_result = run_comparison(entities)
            result = build_comparison_response(entities, comp_result)
        else:
            raise HTTPException(status_code=501, detail=f"Query type '{entities.query_type.value}' is not supported.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Query failed for query='{}': {}", request.query, exc)
        raise HTTPException(status_code=500, detail="Search failed. Try again.") from exc

    if request.session_id:
        save_session_entities(request.session_id, entities)

    return result

@app.get("/health")
def health():
    return {"status": "ok"}