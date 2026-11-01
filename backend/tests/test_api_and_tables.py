import json
from pathlib import Path

from fastapi.testclient import TestClient
from rag_api.api.main import app
from rag_api.db import db_session
from rag_api.models import Conversation, Corpus, Document, LLMCache, Message, Run
from rag_api.runner import model_name


def test_fastapi_native_route_tracing_is_disabled():
    assert app._telemetry["tracing"] is False


def test_table_evidence_is_exact_and_has_source_row(local_store):
    folder = Path(local_store) / "kb"
    folder.mkdir()
    path = folder / "forecast.csv"
    path.write_text("Country Name,2026,2027\nSouth Asia,6.3,6.9\nIndia,6.6,7.2\n", encoding="utf-8")
    with db_session() as db:
        db.add(Corpus(id="test-v1", status="ready"))
        db.add(Document(id="doc-1", corpus_id="test-v1", path="kb/forecast.csv", title="Forecast CSV",
                        source_url="https://example.org/forecast.csv", sha256="x" * 64, mime="text/csv", status="ready"))
    with TestClient(app) as client:
        response = client.post("/v1/tables/query", json={"document_id": "doc-1", "operation": "sum",
                                                       "column": "2027"})
    assert response.status_code == 200
    assert response.json()["result"] == "14.1"
    assert [row["row"] for row in response.json()["rows"]] == [2, 3]


def test_compare_api_sends_same_question_to_separate_runners(local_store, monkeypatch):
    with db_session() as db:
        db.add(Corpus(id="test-v1", status="ready"))
    seen = []

    def fake_runner(run_id, conversation_id, branch, corpus_id):
        seen.append((run_id, conversation_id, branch, corpus_id))
        return {"status": "ok", "answer": f"{branch} response", "citations": [],
                "confidence": {"score": 0.0, "method": "evidence_heuristic_v1"}}

    monkeypatch.setattr("rag_api.api.conversations.run_branch", fake_runner)
    with TestClient(app) as client:
        assert client.get("/v1/corpora/test-v1").json()["status"] == "ready"
        created = client.post("/v1/conversations", json={"corpus_id": "test-v1"})
        assert created.status_code == 200
        conversation_id = created.json()["conversation_id"]
        response = client.post(f"/v1/conversations/{conversation_id}/turns",
                               json={"question": "Aur 2027 mein?"})
        assert response.status_code == 200
        assert set(response.json()["results"]) == {"slm", "mistral"}
        history = client.get(f"/v1/conversations/{conversation_id}").json()["branches"]
    assert {item[2] for item in seen} == {"slm", "mistral"}
    assert history["slm"][0]["content"] == history["mistral"][0]["content"] == "Aur 2027 mein?"


def test_chat_sidebar_lists_saved_sessions_with_first_question(local_store, monkeypatch):
    with db_session() as db:
        db.add(Corpus(id="test-v1", status="ready"))
    monkeypatch.setattr("rag_api.api.conversations.run_branch", lambda run_id, conversation_id, branch, corpus_id:
                        {"status": "ok", "answer": "answer"})
    with TestClient(app) as client:
        conversation_id = client.post("/v1/conversations", json={"corpus_id": "test-v1"}).json()["conversation_id"]
        for _ in range(9):
            client.post("/v1/conversations", json={"corpus_id": "test-v1"})
        initial = client.get("/v1/conversations").json()
        assert initial == []
        client.post(f"/v1/conversations/{conversation_id}/turns", json={"question": "India growth in 2027?"})
        saved = client.get("/v1/conversations").json()
        assert len(saved) == 1
        assert saved[0]["conversation_id"] == conversation_id
        assert saved[0]["title"] == "India growth in 2027?"


def test_delete_chat_removes_history_runs_and_owned_cache(local_store):
    with db_session() as db:
        db.add(Corpus(id="test-v1", status="ready"))
        conversation = Conversation(corpus_id="test-v1")
        db.add(conversation)
        db.flush()
        conversation_id = conversation.id
        db.add(Message(conversation_id=conversation_id, branch="slm", turn_no=1,
                       role="user", content="Question"))
        db.add(Run(id="delete-run", conversation_id=conversation_id, branch="slm", turn_no=1,
                   status="ok", answer="Answer"))
        db.add(LLMCache(key="a" * 64, source_conversation_id=conversation_id,
                        corpus_id="test-v1", branch="slm", model="local-qwen3-4b", answer="Answer"))
    with TestClient(app) as client:
        assert client.get("/v1/conversations").json()[0]["conversation_id"] == conversation_id
        response = client.delete(f"/v1/conversations/{conversation_id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": conversation_id}
        assert client.get("/v1/conversations").json() == []
        assert client.get(f"/v1/conversations/{conversation_id}").status_code == 404
    with db_session() as db:
        assert db.get(LLMCache, "a" * 64) is None
        assert db.get(Run, "delete-run") is None


def test_stream_turn_emits_branch_progress_tokens_and_results(local_store, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from rag_api import observability

    class Collector(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    collector = Collector()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collector))
    monkeypatch.setattr(observability, "TRACER", provider.get_tracer("test"))
    with db_session() as db:
        db.add(Corpus(id="test-v1", status="ready"))

    def fake_runner(run_id, conversation_id, branch, corpus_id, on_event, stream):
        assert stream is True
        with observability.observed(f"agent.{model_name(branch)}", "AGENT", "2027 growth?") as branch_span:
            with observability.observed(f"llm.{model_name(branch)}.call_1", "LLM", "2027 growth?") as llm_span:
                observability.record_output(llm_span, f"{branch} answer")
            observability.record_output(branch_span, f"{branch} answer")
        on_event({"type": "step", "branch": branch, "step": "search", "status": "done"})
        on_event({"type": "token", "branch": branch, "delta": f"{branch} answer"})
        return {"status": "ok", "run_id": run_id, "answer": f"{branch} answer",
                "citations": [], "confidence": {"score": 0.5}}

    monkeypatch.setattr("rag_api.api.conversations.run_branch", fake_runner)
    with TestClient(app) as client:
        conversation_id = client.post("/v1/conversations", json={"corpus_id": "test-v1"}).json()["conversation_id"]
        response = client.post(f"/v1/conversations/{conversation_id}/turns/stream",
                               json={"question": "2027 growth?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "started"
    assert {event["branch"] for event in events if event["type"] == "token"} == {"slm", "mistral"}
    assert {event["branch"] for event in events if event["type"] == "result"} == {"slm", "mistral"}
    assert events[-1]["type"] == "complete"
    spans = collector.spans
    root = next(span for span in spans if span.name == "rag.compare")
    branch_names = {f"agent.{model_name(branch)}" for branch in ("slm", "mistral")}
    model_names = {f"llm.{model_name(branch)}.call_1" for branch in ("slm", "mistral")}
    assert {span.name for span in spans} == {"rag.compare"} | branch_names | model_names
    assert all(span.context.trace_id == root.context.trace_id for span in spans)
    assert all(span.attributes["session.id"] == conversation_id for span in spans)
    assert all(span.attributes["input.value"] == "2027 growth?" for span in spans)
    assert all(span.attributes["input.mime_type"] == "text/plain" for span in spans)
    assert all(span.parent.span_id == root.context.span_id for span in spans if span.name in branch_names)


def test_single_branch_endpoint_does_not_write_other_history(local_store, monkeypatch):
    with db_session() as db:
        db.add(Corpus(id="test-v1", status="ready"))
    monkeypatch.setattr("rag_api.api.conversations.run_branch", lambda run_id, conversation_id, branch, corpus_id:
                        {"status": "ok", "answer": "debug"})
    with TestClient(app) as client:
        conversation_id = client.post("/v1/conversations", json={"corpus_id": "test-v1"}).json()["conversation_id"]
        response = client.post("/v1/branches/slm/turns",
                               json={"conversation_id": conversation_id, "question": "2026?"})
        history = client.get(f"/v1/conversations/{conversation_id}").json()["branches"]
    assert response.status_code == 200
    assert response.json()["branch"] == "slm"
    assert len(history["slm"]) == 1
    assert history["mistral"] == []
