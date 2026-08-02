import json
import os

from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, model_validator

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_EXTRACTION_ATTEMPTS = 2

GENERIC_PROPERTY_TYPE_TERMS = {
    "apartment", "apartments", "house", "houses", "place", "property",
    "properties", "home", "homes", "accommodation", "somewhere",
}

SYSTEM_PROMPT = """You extract structured search intent from natural language queries about renting or buying property in Lagos, Nigeria.

You may be given the entities extracted from the user's PREVIOUS query in this conversation, as
context. If provided, treat the new query as a possible follow-up: any field the new query does
NOT mention should carry over unchanged from the previous entities. Only override a field when
the new query explicitly states something different for it. If the new query changes the topic
entirely (a clearly unrelated request), ignore the previous context and extract fresh from
scratch. If no previous context is provided, extract fresh from scratch as normal.

Return ONLY a JSON object with these fields, no other text, no markdown fences:
- query_type: one of "lookup", "affordability", "comparison", "advice"
  - "lookup" = user wants listings matching specific criteria (beds, area)
  - "affordability" = user gives a budget and wants to know what they can get, optionally where
  - "comparison" = user wants two or more specific options compared against each other
- beds: integer number of bedrooms, or null if not stated (not used when query_type is "comparison")
- area: the Lagos area/neighborhood name as stated by the user, or null if not stated (not used when query_type is "comparison")
- budget_ngn: the budget in Nigerian Naira as a number, or null if not stated
- budget_period: "year" or "month".
  Rent in Lagos is almost always quoted and paid annually. Default to "year" unless the user
  explicitly signals a monthly figure.
- budget_period_was_explicit: true only if the user explicitly said "monthly", "per month", "yearly", "per annum", "annual", or similar. Otherwise false
- property_type: ONLY set this when the user names a specific structural type — "flat", "duplex",
  "self contain", "mini flat", "terrace", "bungalow", "detached house", "semi-detached", "penthouse",
  or similar. Generic words like "apartment", "house", "place", "property", or "somewhere to live" are
  NOT property types — leave this null for those, since they don't map to any real filter.
- listing_type: "rent" or "sale", or null if not stated (not used when query_type is "comparison")
- comparison_options: REQUIRED when query_type is "comparison", otherwise null. A list of 2 or more
  objects, one per option the user wants compared, each with:
  - label: short human description of the option as the user described it (e.g. "2 bedroom on the mainland")
  - beds, area, property_type, listing_type: same meaning as above, per this specific option, null if not stated
  Note: the database has no "mainland" or "island" column. If the user names a region rather than
  a specific area (e.g. "the mainland", "the island"), leave area null for that option and only
  extract beds/property_type/listing_type. Do not guess a specific area for a region-level term.
- "advice" = user is asking a general question about renting/buying in Lagos that isn't about
    specific listings — rent negotiation, agent fees, deposits, tenancy law/practices, road
    conditions, proximity concerns, electricity reliability, red flags to watch for, or similar.
    This does not need beds/area/budget/property_type/listing_type to be set at all.
- resolved_query: REQUIRED when query_type is "advice", otherwise null. A fully self-contained
  rewrite of the user's question, suitable for a web search on its own with no other context.
  If this question is a follow-up to a PREVIOUS advice question (previous context will be given
  to you when applicable), incorporate the prior topic into the rewrite so it stands alone — do
  NOT just repeat the raw follow-up text verbatim. Always mention Lagos/Nigeria explicitly in the
  rewrite if the topic is location-relevant, since the raw follow-up often omits it.
  
Example 1 - lookup:
Query: "I want to rent a 2 bedroom in Oshodi"
Output: {"query_type": "lookup", "beds": 2, "area": "Oshodi", "budget_ngn": null, "budget_period": "year", "budget_period_was_explicit": false, "property_type": null, "listing_type": "rent", "comparison_options": null, "resolved_query": null}

Example 2 - affordability:
Query: "I have 500k, what apartments can I afford and where in Lagos?"
Output: {"query_type": "affordability", "beds": null, "area": null, "budget_ngn": 500000, "budget_period": "year", "budget_period_was_explicit": false, "property_type": "apartment", "listing_type": null, "comparison_options": null, "resolved_query": null}

Example 2b - affordability with generic word:
Query: "I have 800k, what house can I get in Surulere?"
Output: {"query_type": "affordability", "beds": null, "area": "Surulere", "budget_ngn": 800000, "budget_period": "year", "budget_period_was_explicit": false, "property_type": null, "listing_type": null, "comparison_options": null, "resolved_query": null}

Example 3 - comparison:
Query: "I have 350k, should I get a 2 bedroom on the mainland or a single room on the island?"
Output: {"query_type": "comparison", "beds": null, "area": null, "budget_ngn": 350000, "budget_period": "year", "budget_period_was_explicit": false, "property_type": null, "listing_type": null, "comparison_options": [{"label": "2 bedroom on the mainland", "beds": 2, "area": null, "property_type": null, "listing_type": null}, {"label": "single room on the island", "beds": null, "area": null, "property_type": "self contain", "listing_type": null}]}

Example 4 - advice:
Query: "Is agent fee negotiable in Lagos?"
Output: {"query_type": "advice", "beds": null, "area": null, "budget_ngn": null, "budget_period": "year", "budget_period_was_explicit": false, "property_type": null, "listing_type": null, "comparison_options": null, "resolved_query": "Is real estate agent fee negotiable when renting property in Lagos, Nigeria?"}

Example 5 - advice follow-up:
Previous context: {"query_type": "advice", "resolved_query": "How much security deposit is normal for renting an apartment in Lagos, Nigeria?", ...}
Query: "what about for a shared apartment"
Output: {"query_type": "advice", "beds": null, "area": null, "budget_ngn": null, "budget_period": "year", "budget_period_was_explicit": false, "property_type": null, "listing_type": null, "comparison_options": null, "resolved_query": "How much security deposit is normal for renting a shared/roommate apartment in Lagos, Nigeria, compared to renting alone?"}
"""
class QueryType(str, Enum):
    LOOKUP = "lookup"
    AFFORDABILITY = "affordability"
    COMPARISON = "comparison"
    ADVICE = "advice"


class ListingType(str, Enum):
    RENT = "rent"
    SALE = "sale"

class ComparisonOption(BaseModel):
    label: str
    beds: Optional[int] = Field(default=None, ge=0)
    area: Optional[str] = Field(default=None)
    property_type: Optional[str] = Field(default=None)
    listing_type: Optional[ListingType] = Field(default=None)

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
    comparison_options: Optional[list[ComparisonOption]] = Field(default=None)
    resolved_query: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_comparison_options(self):
        if self.query_type == QueryType.COMPARISON and (
            self.comparison_options is None or len(self.comparison_options) < 2
        ):
            raise ValueError("comparison query_type requires at least 2 comparison_options")
        if self.query_type == QueryType.ADVICE and not self.resolved_query:
            raise ValueError("advice query_type requires resolved_query to be set")
        return self

class EntityExtractionError(Exception):
    pass


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EntityExtractionError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)

def _build_follow_up_context(previous: Optional["PropertyEntities"]) -> Optional[str]:
    if previous is None:
        return None
    return (
        "Previous turn's extracted entities, for follow-up merging:\n"
        f"{json.dumps(previous.model_dump(mode='json'))}"
    )

def _call_groq(
    client: Groq,
    user_query: str,
    retry_hint: Optional[str] = None,
    follow_up_context: Optional[str] = None,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if follow_up_context:
        messages.append({"role": "system", "content": follow_up_context})
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

def extract_entities(
    user_query: str,
    previous_entities: Optional[PropertyEntities] = None,
) -> PropertyEntities:
    if not user_query or not user_query.strip():
        raise EntityExtractionError("query text is empty")

    client = _get_client()
    follow_up_context = _build_follow_up_context(previous_entities)
    retry_hint = None
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        try:
            raw = _call_groq(client, user_query, retry_hint, follow_up_context)
            data = json.loads(raw)
            entities = PropertyEntities.model_validate(data)
            entities = _sanitize_property_type(entities)
            logger.info("Extracted entities on attempt {}: {}", attempt, entities.model_dump())
            return entities
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("Extraction attempt {} failed for query='{}': {}", attempt, user_query, exc)
            retry_hint = f"Your previous response was invalid: {exc}. Return valid JSON matching the schema exactly."

    raise EntityExtractionError(f"failed to extract valid entities after {MAX_EXTRACTION_ATTEMPTS} attempts") from last_error

def _sanitize_property_type(entities: PropertyEntities) -> PropertyEntities:
    if entities.property_type and entities.property_type.strip().lower() in GENERIC_PROPERTY_TYPE_TERMS:
        logger.info("Dropping generic property_type '{}' from extraction", entities.property_type)
        entities = entities.model_copy(update={"property_type": None})

    if entities.comparison_options:
        cleaned_options = []
        for option in entities.comparison_options:
            if option.property_type and option.property_type.strip().lower() in GENERIC_PROPERTY_TYPE_TERMS:
                logger.info("Dropping generic property_type '{}' from comparison option '{}'", option.property_type, option.label)
                option = option.model_copy(update={"property_type": None})
            cleaned_options.append(option)
        entities = entities.model_copy(update={"comparison_options": cleaned_options})

    return entities