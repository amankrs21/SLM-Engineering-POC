# Small gold set for the comparison demo

[`gold.jsonl`](gold.jsonl) contains 16 questions from the downloaded World Bank June 2026 corpus. Three groups contain follow-up turns that reuse one conversation ID; other groups each start a fresh conversation. Expected numeric terms were checked against `../data/gep_2026_june/raw/GEPCSV.csv` or the chapter highlights.

After indexing the corpus and starting the API:

```bash
uv run --project backend python scripts/run_eval.py --limit 2
uv run --project backend python scripts/run_eval.py --output eval/results.csv
```

The default corpus is `gep-2026-06-jina-v5`; pass `--corpus gep-2026-06-v1` to compare the older local-Qwen embedding index. The CSV has one row per branch per question: answer, status, citation IDs, expected term/source string checks, heuristic confidence, latency and token counts. `terms_found` and `source_found` are **screening checks**, not a correctness score. Fill `human_correctness` and `human_source_support` after reading the cited passages. For the no-evidence case, check abstention manually. Keep the generated results out of version control unless you intentionally publish a benchmark with model versions, hardware and provider settings.
