import json
from pathlib import Path

from groq import Groq
from loguru import logger

from advice.advice_engine import answer_advice_query
from advice.search_client import search_advice
from config import GROQ_API_KEY, GROQ_MODEL

ADVICE_EVAL_QUERIES_PATH = Path(__file__).parent / "advice_eval_queries.json"
REPORT_PATH = Path(__file__).parent / "reports" / "advice_eval_report.json"


def _load_queries() -> list[dict]:
    with open(ADVICE_EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _judge_score(client: Groq, system_prompt: str, user_content: str) -> float:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Judge returned unparseable score, defaulting to 0.0")
        return 0.0


FAITHFULNESS_PROMPT = """You check whether an answer's claims are actually supported by the given
source context. Return ONLY JSON: {"score": <float 0.0-1.0>} where 1.0 means every claim in the
answer is directly supported by the context, and 0.0 means the answer contains claims not found
in the context at all. No other text."""

RELEVANCY_PROMPT = """You check whether an answer actually addresses the question asked. Return
ONLY JSON: {"score": <float 0.0-1.0>} where 1.0 means the answer directly and fully addresses the
question, and 0.0 means it's off-topic or non-responsive. No other text."""

CONTEXT_PRECISION_PROMPT = """You check what fraction of the given search result snippets are
actually relevant and useful for answering the question. Return ONLY JSON:
{"score": <float 0.0-1.0>} where 1.0 means all snippets are relevant, 0.0 means none are.
No other text."""

CONTEXT_RECALL_PROMPT = """You check whether the given search result snippets, taken together,
contain enough information to support the expected ground-truth answer below. Return ONLY JSON:
{"score": <float 0.0-1.0>} where 1.0 means the snippets fully cover the ground truth, 0.0 means
they cover none of it. No other text."""


def evaluate_one(client: Groq, case: dict) -> dict:
    query = case["query"]
    ground_truth = case.get("ground_truth", "")

    search_results = search_advice(query)
    context_text = "\n\n".join(f"[{r['title']}] {r['content'][:1000]}" for r in search_results)

    advice_result = answer_advice_query(query)
    answer = advice_result["answer"]

    faithfulness = _judge_score(
        client, FAITHFULNESS_PROMPT,
        f"Context:\n{context_text}\n\nAnswer:\n{answer}",
    )
    relevancy = _judge_score(
        client, RELEVANCY_PROMPT,
        f"Question:\n{query}\n\nAnswer:\n{answer}",
    )
    context_precision = _judge_score(
        client, CONTEXT_PRECISION_PROMPT,
        f"Question:\n{query}\n\nSearch result snippets:\n{context_text}",
    )
    context_recall = _judge_score(
        client, CONTEXT_RECALL_PROMPT,
        f"Ground truth:\n{ground_truth}\n\nSearch result snippets:\n{context_text}",
    )

    return {
        "query": query,
        "answer": answer,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def run_eval() -> dict:
    cases = _load_queries()
    client = Groq(api_key=GROQ_API_KEY)
    results = []

    for case in cases:
        result = evaluate_one(client, case)
        results.append(result)
        logger.info(
            "'{}' — faithfulness={} relevancy={} precision={} recall={}",
            result["query"], result["faithfulness"], result["answer_relevancy"],
            result["context_precision"], result["context_recall"],
        )

    def _avg(key: str) -> float:
        return round(sum(r[key] for r in results) / len(results), 3) if results else 0.0

    summary = {
        "total_cases": len(results),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevancy": _avg("answer_relevancy"),
        "avg_context_precision": _avg("context_precision"),
        "avg_context_recall": _avg("context_recall"),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Advice eval complete: {}", {k: v for k, v in summary.items() if k != "results"})
    return summary


if __name__ == "__main__":
    run_eval()