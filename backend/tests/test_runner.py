import json
from types import SimpleNamespace

import pytest
from rag_api import agent_tools, runner
from rag_api.db import db_session
from rag_api.models import Conversation, Corpus, LLMCache, Message, Run
from rag_api.retrieval import Evidence
from sqlalchemy import select


def prepare(question="Hello, how can you help me with?", branches=("slm",), suffix="one"):
    with db_session() as db:
        if db.get(Corpus, "test-v1") is None:
            db.add(Corpus(id="test-v1", status="ready"))
        conversation = Conversation(corpus_id="test-v1")
        db.add(conversation)
        db.flush()
        cid = conversation.id
        for branch in branches:
            db.add(Message(conversation_id=cid, branch=branch, turn_no=1, role="user", content=question))
            db.add(Run(id=f"{branch}-{suffix}", conversation_id=cid, branch=branch, turn_no=1))
    return cid


def tool(name="search_knowledge_base", arguments=None, call_id="call12345"):
    return runner.Generation("", 25, 10, tool_calls=(runner.ToolCall(
        call_id, name, json.dumps(arguments if arguments is not None else {"query": "South Asia 2026 growth"})),))


def test_greeting_goes_to_model_without_retrieval_and_has_no_evidence_percentage(local_store, monkeypatch):
    cid = prepare(branches=("slm", "mistral"))
    prompts = []

    def model(branch, messages, **kwargs):
        prompts.append((branch, messages))
        assert kwargs["tools"]
        assert messages[-1]["content"] == "Hello, how can you help me with?"
        assert kwargs["tool_choice"] == "required"
        return tool("respond_directly", {"intent": "greeting",
                                        "answer": "Hello! I can search documents and compare spreadsheet forecasts."})

    monkeypatch.setattr(runner, "generate", model)
    monkeypatch.setattr(runner, "execute_tool", lambda *args, **kwargs: pytest.fail("Greeting must not invoke tools"))
    for branch in ("slm", "mistral"):
        events = []
        result = runner.run_branch(f"{branch}-one", cid, branch, "test-v1", events.append)
        assert result["status"] == "ok"
        assert result["citations"] == []
        assert result["confidence"]["score"] is None
        assert result["metrics"]["llm_calls"] == 1
        assert result["metrics"]["tools_used"] == []
        assert events[0]["model"] == runner.model_name(branch)
    assert {branch for branch, _ in prompts} == {"slm", "mistral"}


def test_follow_up_model_sees_only_own_branch_and_selects_own_query(local_store, monkeypatch):
    cid = prepare("South Asia growth 2026?", branches=("slm", "mistral"))
    with db_session() as db:
        for branch in ("slm", "mistral"):
            db.add(Message(conversation_id=cid, branch=branch, turn_no=1, role="assistant",
                           content=f"{branch} said South Asia 2026: 6.3% [1]"))
            db.add(Message(conversation_id=cid, branch=branch, turn_no=2, role="user", content="Aur agle saal?"))
            db.add(Run(id=f"{branch}-two", conversation_id=cid, branch=branch, turn_no=2))
    queries = []

    def retrieve(query, corpus_id, **kwargs):
        queries.append(query)
        return [Evidence("chunk1", "South Asia 2027 growth is 6.9%", "forecast.csv", "row 9", None, 0.9, 4.0)]

    def model(branch, messages, **kwargs):
        if messages[-1]["role"] == "user":
            contents = " ".join(item.get("content") or "" for item in messages[1:])
            assert f"{branch} said" in contents
            assert f"{'mistral' if branch == 'slm' else 'slm'} said" not in contents
            return tool(arguments={"query": f"South Asia 2027 growth forecast {branch}"})
        assert messages[-1]["role"] == "tool"
        return runner.Generation("South Asia 2027: 6.9% [1]", 150, 15)

    monkeypatch.setattr(agent_tools, "retrieve", retrieve)
    monkeypatch.setattr(runner, "generate", model)
    for branch in ("slm", "mistral"):
        result = runner.run_branch(f"{branch}-two", cid, branch, "test-v1")
        assert result["status"] == "ok"
        assert result["citations"][0]["location"] == "row 9"
        assert result["metrics"]["llm_calls"] == 2
    assert len(queries) == 2 and all("2027" in query for query in queries)


def test_model_can_retry_insufficient_kb_evidence_and_trace_survives_reload(local_store, monkeypatch):
    cid = prepare("South Asia 2026 GDP growth?")
    queries = []

    def retrieve(query, corpus_id, **kwargs):
        queries.append(query)
        return [] if len(queries) == 1 else [
            Evidence("found", "South Asia: 6.3% in 2026", "report.pdf", "page 50", None, 0.9, 4.0)]

    responses = iter([tool(arguments={"query": "vague query"}),
                      tool(arguments={"query": "South Asia GDP growth 2026"}, call_id="call67890"),
                      runner.Generation("South Asia growth is 6.3% in 2026 [1].", 90, 20)])
    monkeypatch.setattr(agent_tools, "retrieve", retrieve)
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: next(responses))
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert queries == ["vague query", "South Asia GDP growth 2026"]
    assert result["metrics"]["llm_calls"] == 3
    assert result["citations"][0]["ref"] == 1
    with db_session() as db:
        saved = db.get(Run, "slm-one")
        calls = [event for event in saved.metrics["trace"] if event["step"] == "tool" and event["status"] == "done"]
        assert len(calls) == 2
        assert "South Asia" in calls[1]["detail"]["input"]
        assert calls[1]["detail"]["output"]["evidence"][0]["source"] == "report.pdf"


def test_identical_first_turns_reuse_model_cache_but_do_not_share_branch_cache(local_store, monkeypatch):
    first = prepare(suffix="first")
    second = prepare(suffix="second")
    model_calls = []

    def model(branch, messages, **kwargs):
        model_calls.append(branch)
        return tool("respond_directly", {"intent": "greeting",
                                        "answer": "Hello! I can help explore your knowledge base."})

    monkeypatch.setattr(runner, "generate", model)
    one = runner.run_branch("slm-first", first, "slm", "test-v1")
    two = runner.run_branch("slm-second", second, "slm", "test-v1")
    assert one["status"] == two["status"] == "ok"
    assert not one["metrics"]["cache_hit"] and two["metrics"]["cache_hit"]
    assert two["metrics"]["prompt_tokens"] == 0
    assert model_calls == ["slm"]
    with db_session() as db:
        assert db.scalar(select(LLMCache)).source_conversation_id == first
    assert runner.generation_cache_key("slm", "a", "test-v1", [], tools=[]) != runner.generation_cache_key(
        "mistral", "a", "test-v1", [], tools=[])
    assert runner.generation_cache_key("slm", "a", "test-v1", [], tools=[]) != runner.generation_cache_key(
        "slm", "a", "test-v1", [], tools=runner.TOOLS)


def test_tool_error_returns_to_model_for_recovery(local_store, monkeypatch):
    cid = prepare("Find a forecast")
    responses = iter([tool(arguments={"query": ""}), runner.Generation("Please specify the country and year.", 20, 10)])
    prompts = []

    def model(branch, messages, **kwargs):
        prompts.append(list(messages))
        return next(responses)

    monkeypatch.setattr(runner, "generate", model)
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert "error" in json.loads(prompts[1][-1]["content"])
    assert result["confidence"]["score"] == 0


def test_tool_budget_stops_loop_and_requests_final_answer(local_store, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "3")
    runner.get_settings.cache_clear()
    cid = prepare("Find missing data")
    choices = []

    def model(branch, messages, **kwargs):
        choices.append(kwargs["tool_choice"])
        if kwargs["tool_choice"] == "none":
            return runner.Generation("I could not verify this from the available documents.", 40, 12)
        return tool(name="list_tables", arguments={})

    monkeypatch.setattr(runner, "generate", model)
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert choices == ["required", "auto", "none"]
    assert result["metrics"]["tools_used"] == ["list_tables", "list_tables"]


def test_invalid_citation_caps_confidence():
    evidence = [Evidence("one", "The value is 6.9", "report.pdf", "page 1", None, 0.8, 4.0)]
    citations = runner.validate_citations("Value is 6.9 [2]", evidence)
    assert citations == []
    assert runner.evidence_confidence("Value is 6.9 [2]", evidence, citations)["score"] <= 0.3


def test_confidence_accepts_sql_float_noise_but_rejects_a_different_number():
    evidence = [Evidence("one", "2027: average 7.050000000000001", "data.csv", "SQL result", None, 1, 4)]
    supported = "2027 average: 7.05% [1]"
    unsupported = "2027 average: 7.5% [1]"
    assert runner.evidence_confidence(supported, evidence, runner.validate_citations(supported, evidence))[
        "components"]["value_consistency"] == 1
    assert runner.evidence_confidence(unsupported, evidence, runner.validate_citations(unsupported, evidence))[
        "score"] <= 0.2


def test_model_repairs_missing_citations_without_retrieving_again(local_store, monkeypatch):
    cid = prepare("South Asia 2026 growth?")
    monkeypatch.setattr(agent_tools, "retrieve", lambda *args, **kwargs: [
        Evidence("one", "South Asia 2026: 6.3%", "report.pdf", "page 50", None, 0.9, 4.0)])
    responses = iter([tool(), runner.Generation("South Asia: 6.3%", 30, 12),
                      runner.Generation("South Asia: 6.3% [1]", 35, 15)])
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: next(responses))
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert result["answer"] == "South Asia: 6.3% [1]"
    assert result["metrics"]["citation_repairs"] == 1
    assert result["metrics"]["llm_calls"] == 3
    assert result["metrics"]["tools_used"] == ["search_knowledge_base"]


def test_context_compaction_keeps_evidence_citation_ids():
    messages = [{"role": "tool", "content": json.dumps({"evidence": [
        {"ref": index, "citation": f"[{index}]", "text": "source " * 3000}
        for index in range(1, 6)]})}]
    runner.compact_tool_history(messages)
    assert [item["citation"] for item in json.loads(messages[0]["content"])["evidence"]] == ["[1]", "[2]", "[3]", "[4]", "[5]"]
    assert len(json.dumps(messages)) < 32000


def test_missing_mistral_key_fails_before_tools(local_store, monkeypatch):
    cid = prepare(branches=("mistral",))
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    runner.get_settings.cache_clear()
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: pytest.fail("Missing key must fail before model"))
    result = runner.run_branch("mistral-one", cid, "mistral", "test-v1")
    assert result["status"] == "error" and "MISTRAL_API_KEY" in result["error"]


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return iter(self.chunks)

    def __exit__(self, *args):
        pass


def mock_client(monkeypatch, response):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append(kwargs)
            return response

        def close(self):
            pass

    monkeypatch.setattr(runner, "OpenAI", Client)
    return calls


def test_mistral_adapter_accepts_native_tool_calls_and_greedy_sampling(local_store, monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, tool_calls=[SimpleNamespace(id="call12345", function=SimpleNamespace(
            name="search_knowledge_base", arguments='{"query":"South Asia 2026"}'))]))],
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=8))
    calls = mock_client(monkeypatch, response)
    result = runner.generate("mistral", [{"role": "user", "content": "Question"}], tools=runner.TOOLS)
    assert calls[0]["model"] == "mistral-large-4"
    assert calls[0]["top_p"] == 1 and calls[0]["temperature"] == 0
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["parallel_tool_calls"] is False
    assert calls[0]["tool_choice"] == "auto"
    assert result.tool_calls[0].name == "search_knowledge_base"
    assert result.prompt_tokens == 42


def test_stream_emits_visible_answer_and_omits_private_thinking(local_store, monkeypatch):
    calls = mock_client(monkeypatch, FakeStream([
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=[{"type": "thinking", "thinking": "hidden"}]))], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=[{"type": "text", "text": "6.3% [1]"}]))], usage=None),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=40, completion_tokens=12)),
    ]))
    tokens = []
    result = runner.generate("mistral", [], on_token=tokens.append)
    assert tokens == ["6.3% [1]"]
    assert result == runner.Generation("6.3% [1]", 40, 12)
    assert calls[0]["stream"] is True


def test_stream_assembles_tool_argument_deltas_and_disables_qwen_thinking(local_store, monkeypatch):
    def delta(arguments, name=None, call_id=None):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[
            SimpleNamespace(index=0, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
        ]))], usage=None)

    calls = mock_client(monkeypatch, FakeStream([
        delta('{"que', "search_knowledge_base", "call12345"), delta('ry":"South Asia"}'),
    ]))
    tokens = []
    result = runner.generate("slm", [], on_token=tokens.append, tools=runner.TOOLS)
    assert tokens == []
    assert result.tool_calls == (runner.ToolCall("call12345", "search_knowledge_base", '{"query":"South Asia"}'),)
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_empty_visible_stream_is_error(local_store, monkeypatch):
    mock_client(monkeypatch, FakeStream([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=[{"type": "thinking", "thinking": "hidden"}]))], usage=None)]))
    with pytest.raises(RuntimeError, match="no visible answer"):
        runner.generate("mistral", [], on_token=lambda token: None)


def test_phoenix_spans_have_actual_model_name_session_and_each_model_tool_round(local_store, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from rag_api import observability

    class Exporter(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

    exporter = Exporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "TRACER", provider.get_tracer("test"))
    cid = prepare("India 2027 growth?")
    monkeypatch.setattr(agent_tools, "retrieve", lambda *args, **kwargs: [
        Evidence("one", "India 2027: 7.2%", "forecast.csv", "row 3", None, 0.9, 4.0)])
    responses = iter([tool(arguments={"query": "India 2027 growth"}), runner.Generation("7.2% [1]", 42, 8)])
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: next(responses))
    assert runner.run_branch("slm-one", cid, "slm", "test-v1")["status"] == "ok"
    spans = {span.name: span for span in exporter.spans}
    name = runner.model_name("slm")
    assert {f"agent.{name}", f"llm.{name}.call_1", f"llm.{name}.call_2", "tool.search_knowledge_base"} <= spans.keys()
    assert all(span.attributes["session.id"] == cid for span in exporter.spans)
    final = spans[f"llm.{name}.call_2"]
    assert final.attributes["input.value"] == "India 2027 growth?"
    assert final.attributes["output.value"] == "7.2% [1]"
    assert final.attributes["llm.token_count.total"] == 50
    assert final.attributes["llm.model_name"] == name
    assert "forecast.csv" in spans["tool.search_knowledge_base"].attributes["output.value"]
    assert not any(value == "system" for key, value in final.attributes.items() if key.endswith(".message.role"))


def test_cached_tool_decisions_still_execute_tools_against_current_corpus(local_store, monkeypatch):
    first = prepare("South Asia 2026?", suffix="first")
    second = prepare("South Asia 2026?", suffix="second")
    model_calls = []
    searches = []

    def retrieve(query, corpus_id, **kwargs):
        searches.append(query)
        return [Evidence("one", "South Asia 2026: 6.3%", "data.csv", "row 9", None, 0.9, 4)]

    def model(branch, messages, **kwargs):
        model_calls.append(branch)
        return tool() if messages[-1]["role"] == "user" else runner.Generation("6.3% [1]", 30, 12)

    monkeypatch.setattr(agent_tools, "retrieve", retrieve)
    monkeypatch.setattr(runner, "generate", model)
    one = runner.run_branch("slm-first", first, "slm", "test-v1")
    two = runner.run_branch("slm-second", second, "slm", "test-v1")
    assert one["status"] == two["status"] == "ok"
    assert len(model_calls) == 2 and len(searches) == 2
    assert two["metrics"]["cache_hits"] == 2
    assert two["citations"][0]["source"] == "data.csv"


def test_historical_citation_without_current_evidence_requires_model_to_fetch_sources(local_store, monkeypatch):
    cid = prepare("South Asia 2027?")
    monkeypatch.setattr(agent_tools, "retrieve", lambda *args, **kwargs: [
        Evidence("one", "South Asia 2027: 6.9%", "data.csv", "row 9", None, 1, 4)])
    responses = iter([tool("list_tables", {}), runner.Generation("6.9% [1]", 30, 10),
                      tool(arguments={"query": "South Asia 2027"}),
                      runner.Generation("South Asia 2027: 6.9% [1]", 35, 15)])
    prompts = []

    def model(branch, messages, **kwargs):
        prompts.append(list(messages))
        return next(responses)

    monkeypatch.setattr(runner, "generate", model)
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert "NO sources" in prompts[2][-1]["content"]
    assert result["citations"][0]["source"] == "data.csv"
    assert result["metrics"]["tools_used"] == ["list_tables", "search_knowledge_base"]


def test_model_cannot_publish_invented_source_citations_as_a_direct_response(local_store, monkeypatch):
    cid = prepare("South Asia 2027?")
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: tool(
        "respond_directly", {"intent": "greeting", "answer": "6.9% [99]"}))
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert "6.9" not in result["answer"] and "[99]" not in result["answer"]
    assert result["metrics"]["grounding_failed"] is True
    assert result["confidence"]["score"] == 0


def test_initial_model_call_must_select_a_structured_action(local_store, monkeypatch):
    cid = prepare("South Asia 2028?")
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: runner.Generation("6.8%", 30, 10))
    result = runner.run_branch("slm-one", cid, "slm", "test-v1")
    assert result["status"] == "error"
    assert "initial action" in result["error"]
