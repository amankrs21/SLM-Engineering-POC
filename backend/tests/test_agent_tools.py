import json

import pytest
from rag_api import agent_tools, runner
from rag_api.db import db_session
from rag_api.models import Conversation, Corpus, Document, Message, Run
from rag_api.table_tools import query_table_sql


def add_table(folder, corpus_id="test-v1", document_id="table1", name="forecast.csv"):
    (folder / name).write_text("Country Name,2026,2027\nSouth Asia,6.3,6.9\nIndia,6.6,7.2\n", encoding="utf-8")
    with db_session() as db:
        if db.get(Corpus, corpus_id) is None:
            db.add(Corpus(id=corpus_id, status="ready"))
        document = Document(id=document_id, corpus_id=corpus_id, path=name, title=name,
                            sha256="demo", mime="text/csv", status="ready")
        db.add(document)
    return document


def test_read_only_sql_uses_real_column_names_and_exact_numeric_result(local_store):
    document = add_table(local_store)
    result = query_table_sql(document, sql='SELECT "2027", _source_row FROM data WHERE "Country Name" = \'South Asia\'')
    assert result["rows"] == [{"2027": 6.9, "_source_row": 2}]
    assert query_table_sql(document, sql='SELECT ROUND(SUM("2027"), 4) AS total FROM data')["rows"] == [{"total": 14.1}]


@pytest.mark.parametrize("sql", [
    "DELETE FROM data", "ATTACH DATABASE '/tmp/evil' AS other", "SELECT * FROM sqlite_master",
    "SELECT load_extension('evil')", "SELECT * FROM data; DELETE FROM data",
    "SELECT randomblob(100)",
    "WITH x AS (SELECT 1) DELETE FROM data", "SELECT * FROM branch_messages",
])
def test_sql_cannot_modify_data_or_access_application_database(local_store, sql):
    document = add_table(local_store)
    before = (local_store / "forecast.csv").read_bytes()
    with pytest.raises(ValueError):
        query_table_sql(document, sql=sql)
    assert (local_store / "forecast.csv").read_bytes() == before


def test_tools_restrict_documents_to_current_corpus(local_store):
    add_table(local_store, corpus_id="other", document_id="other-table")
    with pytest.raises(ValueError, match="does not belong"):
        agent_tools.execute_tool("inspect_table", '{"document_id":"other-table"}', "test-v1", [], {})


def test_model_can_choose_inspection_then_natural_language_to_sql_with_citation(local_store, monkeypatch):
    add_table(local_store)
    with db_session() as db:
        conversation = Conversation(corpus_id="test-v1")
        db.add(conversation)
        db.flush()
        cid = conversation.id
        db.add(Message(conversation_id=cid, branch="slm", turn_no=1, role="user", content="South Asia in 2027?"))
        db.add(Run(id="sql-run", conversation_id=cid, branch="slm", turn_no=1))
    rounds = []

    def model(branch, messages, **kwargs):
        rounds.append(messages[-1])
        if len(rounds) == 1:
            name, args = "list_tables", {}
        elif len(rounds) == 2:
            tables = json.loads(messages[-1]["content"])["tables"]
            name, args = "inspect_table", {"document_id": tables[0]["document_id"]}
        elif len(rounds) == 3:
            schema = json.loads(messages[-1]["content"])
            assert "2027" in schema["columns"] and schema["sample_rows"][0]["Country Name"] == "South Asia"
            name, args = "query_table_sql", {"document_id": "table1", "sql":
                'SELECT "2027" FROM data WHERE "Country Name" = \'South Asia\''}
        else:
            result = json.loads(messages[-1]["content"])
            return runner.Generation(f'South Asia: {result["rows"][0]["2027"]}% [{result["ref"]}]', 30, 12)
        return runner.Generation("", 30, 10, tool_calls=(runner.ToolCall(f"call{len(rounds):05d}", name, json.dumps(args)),))

    monkeypatch.setattr(runner, "generate", model)
    result = runner.run_branch("sql-run", cid, "slm", "test-v1")
    assert result["status"] == "ok"
    assert result["answer"] == "South Asia: 6.9% [1]"
    assert result["metrics"]["tools_used"] == ["list_tables", "inspect_table", "query_table_sql"]
    assert result["citations"][0]["source"] == "forecast.csv"
    assert result["confidence"]["components"]["value_consistency"] == 1


def test_csv_tool_normalizes_invented_sheet_with_explicit_feedback(local_store):
    add_table(local_store)
    args = json.dumps({"document_id": "table1", "sheet": "invented sheet"})
    result = agent_tools.execute_tool("inspect_table", args, "test-v1", [], {})
    assert result["selected_sheet"] is None
    assert "ignored" in result["note"]
    assert result["columns"] == ["Country Name", "2026", "2027"]
