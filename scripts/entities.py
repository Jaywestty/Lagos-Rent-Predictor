import json
import os

from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_EXTRACTION_ATTEMPTS = 2

SYSTEM_PROMPT = """You extract structured search intent from natural language queries about renting or buying property in Lagos, Nigeria.

Return ONLY a JSON object with these fields, no other text, no markdown fences:
- query_type: one of "lookup", "affordability", "comparison"
  - "lookup" = user wants listings matching specific criteria (beds, area)
  - "affordability" = user gives a budget and wants to know what they can get
  - "comparison" = user wants two or more options compared
- beds: integer number of bedrooms, or null if not stated
- area: the Lagos area/neighborhood name as stated by the user, or null if not stated
- budget_ngn: the budget in Nigerian Naira as a number, or null if not stated
- budget_period: "year" or "month". If the user did not explicitly state a period, default to "year"
- budget_period_was_explicit: true only if the user explicitly said "monthly", "per month", "yearly", "per annum", "annual", or similar. Otherwise false
- property_type: the property type as stated (e.g. "flat", "duplex", "self contain", "mini flat"), or null if not stated
- listing_type: "rent" or "sale", or null if not stated

Example query: "I want to rent a 2 bedroom in Oshodi"
Example output: {"query_type": "lookup", "beds": 2, "area": "Oshodi", "budget_ngn": null, "budget_period": "year", "budget_period_was_explicit": false, "property_type": null, "listing_type": "rent"}
"""

class QueryType(str, Enum):
    LOOKUP = "lookup"
    AFFORDABILITY = "affordability"
    COMPARISON = "comparison"


class ListingType(str, Enum):
    RENT = "rent"
    SALE = "sale"


class BudgetPeriod(str, Enum):
    YEAR = "year"
    MONTH = "month"


class PropertyEntities(BaseModel):
    query_type: QueryType
    beds: Optional[int] = Field(default=None, ge=0)
    area: Optional[str] = Field(default=None)
    budget_ngn: Optional[float] = Field(default=None, ge=0)
    budget_period: BudgetPeriod = Field(default=BudgetPeriod.YEAR)
    budget_period_was_explicit: bool = Field(default=False)
    property_type: Optional[str] = Field(default=None)
    listing_type: Optional[ListingType] = Field(default=None)


class EntityExtractionError(Exception):
    pass


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EntityExtractionError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def _call_groq(client: Groq, user_query: str, retry_hint: Optional[str] = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if retry_hint:
        messages.append({"role": "system", "content": retry_hint})
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


def extract_entities(user_query: str) -> PropertyEntities:
    if not user_query or not user_query.strip():
        raise EntityExtractionError("query text is empty")

    client = _get_client()
    retry_hint = None
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        try:
            raw = _call_groq(client, user_query, retry_hint)
            data = json.loads(raw)
            entities = PropertyEntities.model_validate(data)
            logger.info("Extracted entities on attempt {}: {}", attempt, entities.model_dump())
            return entities
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("Extraction attempt {} failed for query='{}': {}", attempt, user_query, exc)
            retry_hint = f"Your previous response was invalid: {exc}. Return valid JSON matching the schema exactly."

    raise EntityExtractionError(f"failed to extract valid entities after {MAX_EXTRACTION_ATTEMPTS} attempts") from last_error