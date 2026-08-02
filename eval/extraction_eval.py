import json
from pathlib import Path
from typing import Any

from groq import Groq
from loguru import logger

from config import GROQ_API_KEY, GROQ_MODEL
from query.entities import extract_entities

GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
REPORT_PATH = Path(__file__).parent / "reports" / "extraction_eval_report.json"

STRUCTURED_FIELDS = [
    "query_type", "beds", "area", "budget_ngn", "budget_period",
    "budget_period_was_explicit", "property_type", "listing_type",
]

JUDGE_SYSTEM_PROMPT = """You judge whether a rewritten search question faithfully preserves the
topic and intent of an original question, given any prior conversation context. Respond with ONLY
"yes" or "no", no other text."""


def _load_golden_queries() -> list[dict]:
    with open(GOLDEN_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _judge_resolved_query(original_query: str, resolved_query: str) -> bool:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original question: {original_query}\nRewritten question: {resolved_query}"},
        ],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


def _compare_comparison_options(expected: list[dict], actual: list) -> dict:
    if len(expected) != len(actual):
        return {"match": False, "reason": f"expected {len(expected)} options, got {len(actual)}"}

    mismatches = []
    for i, (exp_opt, act_opt) in enumerate(zip(expected, actual)):
        act_dict = act_opt.model_dump(mode="json") if hasattr(act_opt, "model_dump") else act_opt
        for field, exp_value in exp_opt.items():
            act_value = act_dict.get(field)
            if act_value != exp_value:
                mismatches.append(f"option[{i}].{field}: expected={exp_value!r} actual={act_value!r}")

    return {"match": not mismatches, "mismatches": mismatches}


def evaluate_one(case: dict) -> dict:
    query = case["query"]
    expected = case["expected"]

    try:
        entities = extract_entities(query)
    except Exception as exc:
        return {"query": query, "passed": False, "error": str(exc)}

    actual = entities.model_dump(mode="json")
    field_results = {}
    all_passed = True

    for field in STRUCTURED_FIELDS:
        if field not in expected:
            continue
        exp_value = expected[field]
        act_value = actual.get(field)
        passed = exp_value == act_value
        field_results[field] = {"expected": exp_value, "actual": act_value, "passed": passed}
        if not passed:
            all_passed = False

    if "comparison_options" in expected:
        comp_result = _compare_comparison_options(expected["comparison_options"], entities.comparison_options or [])
        field_results["comparison_options"] = comp_result
        if not comp_result["match"]:
            all_passed = False

    if expected.get("query_type") == "advice":
        if not entities.resolved_query:
            field_results["resolved_query"] = {"passed": False, "reason": "resolved_query missing"}
            all_passed = False
        else:
            judged_ok = _judge_resolved_query(query, entities.resolved_query)
            field_results["resolved_query"] = {
                "resolved_query": entities.resolved_query,
                "passed": judged_ok,
            }
            if not judged_ok:
                all_passed = False

    return {"query": query, "passed": all_passed, "fields": field_results}


def run_eval() -> dict:
    cases = _load_golden_queries()
    results = []

    for case in cases:
        result = evaluate_one(case)
        results.append(result)
        status = "PASS" if result.get("passed") else "FAIL"
        logger.info("[{}] {}", status, case["query"])

    passed_count = sum(1 for r in results if r.get("passed"))
    summary = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "accuracy": round(passed_count / len(results), 3) if results else 0.0,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Extraction eval: {}/{} passed ({}%)", passed_count, len(results), round(summary["accuracy"] * 100, 1))
    return summary


if __name__ == "__main__":
    run_eval()