from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from entities import EntityExtractionError, QueryType, extract_entities
from query_engine import run_lookup
from response import build_lookup_response

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

    if entities.query_type != QueryType.LOOKUP:
        raise HTTPException(
            status_code=501,
            detail=f"Query type '{entities.query_type.value}' is not yet supported.",
        )

    try:
        listings = run_lookup(entities)
    except Exception as exc:
        logger.error("Lookup query failed for query='{}': {}", request.query, exc)
        raise HTTPException(status_code=500, detail="Search failed. Try again.") from exc

    return build_lookup_response(entities, listings)


@app.get("/health")
def health():
    return {"status": "ok"}