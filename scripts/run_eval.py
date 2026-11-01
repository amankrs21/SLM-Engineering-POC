"""Run the same gold questions through both independent API branches."""

import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def has_expected_term(answer: str, term: str) -> bool:
    try:
        expected = Decimal(term)
    except InvalidOperation:
        return term.casefold() in answer.casefold()
    return expected in {Decimal(value) for value in re.findall(r"\b\d+(?:\.\d+)?\b", answer)}


def evaluate(api: str, corpus: str, cases: list[dict]) -> list[dict]:
    rows = []
    conversations: dict[str, str] = {}
    with httpx.Client(base_url=api.rstrip("/"), timeout=600) as client:
        for case in cases:
            group = case["group"]
            if group not in conversations:
                created = client.post("/v1/conversations", json={"corpus_id": corpus})
                created.raise_for_status()
                conversations[group] = created.json()["conversation_id"]
            response = client.post(f"/v1/conversations/{conversations[group]}/turns",
                                   json={"question": case["question"]})
            response.raise_for_status()
            results = response.json()["results"]
            for branch in ("slm", "mistral"):
                result = results[branch]
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                expected_terms = case.get("expected_terms", [])
                expected_source = case.get("expected_source", "")
                rows.append({
                    "case_id": case["id"], "group": group, "branch": branch,
                    "question": case["question"], "status": result["status"], "answer": answer,
                    "expected_terms": ", ".join(expected_terms),
                    "terms_found": all(has_expected_term(answer, term) for term in expected_terms)
                    if expected_terms else "human review",
                    "expected_source": expected_source,
                    "source_found": any(expected_source.casefold() in item["source"].casefold()
                                        for item in citations) if expected_source else "human review",
                    "should_abstain": case.get("should_abstain", False),
                    "citations": json.dumps(citations, ensure_ascii=False),
                    "confidence": result.get("confidence", {}).get("score"),
                    "latency_ms": result.get("metrics", {}).get("total_ms"),
                    "prompt_tokens": result.get("metrics", {}).get("prompt_tokens"),
                    "completion_tokens": result.get("metrics", {}).get("completion_tokens"),
                    "error": result.get("error", ""),
                    "human_correctness": "", "human_source_support": "",
                })
            print(f"{case['id']}: SLM={results['slm']['status']}, Mistral={results['mistral']['status']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run side-by-side RAG evaluation")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus", default="gep-2026-06-jina-v5")
    parser.add_argument("--cases", type=Path, default=Path("eval/gold.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results.csv"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    rows = evaluate(args.api, args.corpus, cases[: args.limit] if args.limit else cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} branch results to {args.output}")


if __name__ == "__main__":
    main()
