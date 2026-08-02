from typing import Optional

from groq import Groq
from loguru import logger

from advice.search_client import SearchError, search_advice
from config import GROQ_API_KEY, GROQ_MODEL

SYNTHESIS_SYSTEM_PROMPT = """You are a helpful assistant answering questions about renting or buying
property in Lagos, Nigeria — things like rent negotiation norms, agent fees, deposits, tenancy
practices, neighborhood characteristics (roads, proximity, electricity availability), and red flags
to watch for.

You will be given a user question and a set of web search results. Answer using ONLY information
grounded in the provided search results. Do not add facts you were not given. If the search results
don't actually answer the question, say so plainly rather than guessing.

Keep the answer concise and practical — a renter reading this should come away knowing what to do
or watch for, not a general essay. Do not fabricate URLs or sources; only refer to the sources you
were given.
"""


class AdviceError(Exception):
    pass


def _get_client() -> Groq:
    if not GROQ_API_KEY:
        raise AdviceError("GROQ_API_KEY is not set")
    return Groq(api_key=GROQ_API_KEY)


def _format_search_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Source {i}] {r['title']}\nURL: {r['url']}\n{r['content'][:1500]}")
    return "\n\n".join(blocks)


def answer_advice_query(user_query: str) -> dict:
    try:
        results = search_advice(user_query)
    except SearchError as exc:
        logger.error("Advice search failed for query='{}': {}", user_query, exc)
        return {
            "query_type": "advice",
            "answer": "I couldn't search for that right now. Please try again shortly.",
            "sources": [],
        }

    if not results:
        logger.warning("No usable search results for advice query='{}'", user_query)
        return {
            "query_type": "advice",
            "answer": "I couldn't find reliable information to answer that. Try rephrasing the question.",
            "sources": [],
        }

    client = _get_client()
    context = _format_search_context(results)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {user_query}\n\nSearch results:\n{context}"},
            ],
            temperature=0.2,
        )
        answer_text = response.choices[0].message.content
    except Exception as exc:
        logger.error("Advice synthesis failed for query='{}': {}", user_query, exc)
        return {
            "query_type": "advice",
            "answer": "I found some information but couldn't put together an answer. Please try again.",
            "sources": [{"title": r["title"], "url": r["url"]} for r in results],
        }

    logger.info("Synthesized advice answer for query='{}' using {} source(s)", user_query, len(results))
    return {
        "query_type": "advice",
        "answer": answer_text,
        "sources": [{"title": r["title"], "url": r["url"]} for r in results],
    }