from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from entities import EntityExtractionError, QueryType, extract_entities
from query_engine import run_lookup, run_affordability, run_comparison
from response import build_lookup_response, build_affordability_response, build_comparison_response

logger.add("logs/app.log", rotation="10 MB", retention="14 days", level="INFO")

app = FastAPI(title="Lagos House Hunter AI")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_listings(request: QueryRequest):
    try:
        entities = extract_entities(request.query)
    except EntityExtractionError as exc:
        logger.error("Entity extraction failed for query='{}': {}", request.query, exc)
        raise HTTPException(status_code=422, detail="Could not understand the query.") from exc

    try:
        if entities.query_type == QueryType.LOOKUP:
            listings = run_lookup(entities)
            return build_lookup_response(entities, listings)
        if entities.query_type == QueryType.AFFORDABILITY:
            result = run_affordability(entities)
            return build_affordability_response(entities, result)
        if entities.query_type == QueryType.COMPARISON:
            result = run_comparison(entities)
            return build_comparison_response(entities, result)
    except Exception as exc:
        logger.error("Query failed for query='{}': {}", request.query, exc)
        raise HTTPException(status_code=500, detail="Search failed. Try again.") from exc

@app.get("/health")
def health():
    return {"status": "ok"}