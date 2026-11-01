# RAG FastAPI backend

`backend/` standalone `uv` Python project hai. FastAPI API, one-shot ingestion aur local embedding/reranker service isi package se alag Compose containers mein run hote hain. UI [`frontend/`](../frontend/) ka static nginx page hai; root par Python project ya `.venv` nahi hai.

```bash
uv sync --project backend
uv run --project backend uvicorn rag_api.api.main:app --host 127.0.0.1 --port 8000
```

## Package map

| Path | Responsibility |
| --- | --- |
| `src/rag_api/api/` | Conversation, streaming, corpus, upload and table endpoints |
| `src/rag_api/runner.py` | Independent model-directed agent loop, native tool-call streaming, citations, confidence |
| `src/rag_api/agent_tools.py` | Validated model tool schemas and corpus-scoped execution |
| `src/rag_api/table_tools.py` | Spreadsheet inspection, table API operations and bounded read-only SQL |
| `src/rag_api/llm_cache.py` | Exact input hash and Postgres-backed generated-answer cache |
| `src/rag_api/model_server.py` | Optional local Qwen embedding/reranker service |
| `src/rag_api/retrieval.py` | Corpus-aware Jina/local embedding, Qdrant search, Jina/Cohere/local rerank |
| `src/rag_api/ingest.py` | One-shot, idempotent folder indexing |
| `src/rag_api/observability.py` | Phoenix `CHAIN`, `TOOL`, `LLM` input/output spans |
| `src/rag_api/db.py`, `models.py` | PostgreSQL chat and metadata |

`POST /v1/conversations/{id}/turns/stream` returns SSE `started`, per-branch `step` and `token`, per-branch `result`, then `complete`. `answer_reset` clears intermediate model text when a tool call or citation repair is requested. Step events include actual model, round and tool-call ID; `metrics.trace` persists the activity for reload. Saved chat remains in Postgres; a disconnected UI can reload it. The non-streaming compare endpoint remains available. `GET /v1/conversations` supplies only sessions with a user message; `POST /v1/conversations` creates a session ID on the first UI send. New Chat in the browser is a local draft until then, with `/chat/new` URL.

`DELETE /v1/conversations/{id}` removes its messages, runs and cache entries it created. UI asks for confirmation first. A running turn returns 409 so a worker cannot write into a deleted session.

LLM cache key includes branch, model, corpus, messages, tool schemas, tool choice, sampling and `llm_cache_version`; cache stores only the hash and serialized public answer/tool calls, not prompt text. Every model round can reuse an exact response; selected tools still execute against current corpus data. On a hit usage counts are 0; Phoenix LLM span carries `llm.cache_hit=true`. `metrics.cache_hits` counts reused rounds; `metrics.cache_hit` is true only when all rounds were cached. Changing a model revision should increment `llm_cache_version`.

Phoenix project `slm-rag-model-runs` traces only model runs. FastAPI native route tracing is disabled. Each paired turn is one `rag.compare` trace with `agent.<actual-model-name>` children and one `llm.<actual-model-name>.call_N` span per model step. Greetings usually use one model step per branch; factual questions can use several. Every span has `session.id` equal to the conversation ID. Root/LLM input is the question; output is the answer or selected tool call. Tool spans contain validated input and result. LLM message attributes include conversation/tool messages, excluding the system prompt and private reasoning. The response's expandable Thinking trace shows these public operations, not internal chain-of-thought.

First model call uses `tool_choice=required`: the model selects a source tool or `respond_directly` for a greeting/capability/thanks/clarification. Source tools: `search_knowledge_base`, `list_tables`, `inspect_table`, `query_table_sql`. After results, `tool_choice=auto` allows the model to answer or continue using tools. SQLite queries operate on a temporary in-memory table from one corpus-scoped document. SELECT authorization, row caps and an execution budget block writes/application DB access. SQLite decimal aggregation uses floating point; use `ROUND` for display. Default limits: 6 model rounds, 8 source-tool calls; final budget round disables tools. Citation feedback allows one revision; unresolved invalid references produce a verification-gap reply. Follow-up interpretation is model-driven from the branch's recent messages; prior citation tokens are removed and previous tool results are not reused as fresh evidence.

The current default corpus uses Jina v5 text embeddings. Existing Qwen corpus is still supported because retrieval selects the embedding provider recorded on each corpus. Changing embedding model requires a new corpus ID and full reindex. Jina rerank 429/HTTP failures can fall back to Cohere; embeddings cannot switch providers without reindexing.
