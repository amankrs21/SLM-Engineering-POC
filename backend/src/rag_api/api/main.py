"""FastAPI entrypoint and observability bootstrap."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_api.api.conversations import router as conversations_router
from rag_api.api.knowledge import router as knowledge_router
from rag_api.db import init_db
from rag_api.settings import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    endpoint = get_settings().phoenix_collector_endpoint
    if endpoint:
        try:
            from phoenix.otel import register

            register(project_name="slm-rag-model-runs", endpoint=endpoint, auto_instrument=False)
        except Exception as exc:  # noqa: BLE001 - tracing must not break chat startup
            print(f"Phoenix tracing unavailable: {exc}")
    yield


app = FastAPI(title="SLM versus Mistral RAG POC", lifespan=lifespan,
              telemetry={"tracing": False, "metrics": False, "logs": False})
app.include_router(conversations_router)
app.include_router(knowledge_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
