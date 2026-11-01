"""Model-selectable tools scoped to the current conversation's knowledge corpus."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from rag_api.db import db_session
from rag_api.models import Document
from rag_api.retrieval import Evidence, retrieve
from rag_api.table_tools import inspect_table, query_table_sql, read_table


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchInput(ToolInput):
    query: str = Field(min_length=1, max_length=1000)


class ListInput(ToolInput):
    pass


class InspectInput(ToolInput):
    document_id: str
    sheet: str | None = Field(default=None, description="CSV: null. Excel: null on first inspection to discover "
                                                      "sheets; later use only a returned sheet name.")
    header_row: int = Field(default=1, ge=1, le=100)


class SQLInput(InspectInput):
    sql: str = Field(min_length=1, max_length=4000)


class DirectInput(ToolInput):
    intent: Literal["greeting", "capabilities", "thanks", "clarification"]
    answer: str = Field(min_length=1, max_length=1600,
                        description="Short social/capability reply or clarification question. "
                                    "Never use this action to answer document questions or economic forecasts.")


TOOL_INPUTS = {"search_knowledge_base": SearchInput, "list_tables": ListInput,
               "inspect_table": InspectInput, "query_table_sql": SQLInput, "respond_directly": DirectInput}
DESCRIPTIONS = {
    "search_knowledge_base": "Search corpus documents. Retrieves top 50, reranks to 5. Returns numbered evidence. "
                             "Use a standalone query resolved from YOUR chat history. If results are insufficient, "
                             "search again with a different query. Do not use for greetings or your capabilities.",
    "list_tables": "List available CSV/Excel documents and IDs in this corpus before inspecting or querying one.",
    "inspect_table": "Get sheet names, columns and 3 sample rows. Change header_row if headers are wrong. "
                     "Inspect before SQL; never invent column names or document IDs.",
    "query_table_sql": "Run a read-only SQLite SELECT on one inspected CSV/Excel sheet. Table name: data. "
                       'Quote column names with double quotes. _source_row contains original row numbers. '
                       "Supports filters, SUM, AVG and comparisons; use ROUND for decimal aggregates. "
                       "Returns up to 25 rows with a citation. "
                       "Never query the app database. Use exact tool numbers in your answer.",
    "respond_directly": "Finish a greeting, thanks, explanation of your own capabilities, or ask a clarification. "
                        "This action is ONLY for non-data replies. For factual document/economic questions and "
                        "year follow-ups, select search_knowledge_base or table tools instead, even if history "
                        "contains an earlier answer. No source citations in this action.",
}
TOOLS = [{"type": "function", "function": {"name": name, "description": DESCRIPTIONS[name],
                                           "parameters": schema.model_json_schema()}}
         for name, schema in TOOL_INPUTS.items()]


def execute_tool(name: str, arguments: str, corpus_id: str, evidence: list[Evidence],
                 retrieval_data: dict, on_step=None) -> dict:
    """Validate model arguments, execute one tool, and assign stable turn-local citations."""
    if name not in TOOL_INPUTS:
        raise ValueError(f"Unknown tool: {name}")
    request = TOOL_INPUTS[name].model_validate_json(arguments)
    if name == "respond_directly":
        return request.model_dump()
    if name == "search_knowledge_base":
        trace_data: dict = {}
        found = retrieve(request.query, corpus_id, trace_data=trace_data, on_step=on_step)
        retrieval_data.setdefault("searches", []).append({"query": request.query, **trace_data})
        rows = []
        for item in found:
            existing = next((index for index, source in enumerate(evidence)
                             if source.chunk_id == item.chunk_id), None)
            if existing is None:
                evidence.append(item)
                existing = len(evidence) - 1
            rows.append({"ref": existing + 1, "citation": f"[{existing + 1}]",
                         "source": item.filename, "location": item.locator,
                         "text": item.text[:1000], "source_url": item.source_url})
        return {"query": request.query, "evidence": rows,
                "note": "Cite the citation token of each source you use. Assess relevance; retry another query "
                        "or inspect tables if insufficient. Never use unrelated evidence."}
    with db_session() as db:
        if name == "list_tables":
            documents = db.scalars(select(Document).where(
                Document.corpus_id == corpus_id, Document.status == "ready"
            ).order_by(Document.path)).all()
            tables = [{"document_id": document.id, "filename": Path(document.path).name,
                       "format": Path(document.path).suffix.lower().lstrip("."), "title": document.title}
                      for document in documents if Path(document.path).suffix.lower() in {".csv", ".xlsx"}]
            return {"tables": tables[:40], "truncated": len(tables) > 40}
        document = db.get(Document, request.document_id)
        if document is None or document.corpus_id != corpus_id or document.status != "ready":
            raise ValueError("Document is not ready or does not belong to this corpus; use list_tables")
        # A CSV has exactly one table; some small models supply an invented sheet.
        # Normalize that in the document adapter and tell the model what happened.
        csv_sheet = Path(document.path).suffix.lower() == ".csv" and request.sheet is not None
        selected_sheet = None if csv_sheet else request.sheet
        if name == "inspect_table":
            result = inspect_table(document, selected_sheet, request.header_row)
            sample = read_table(document, result["selected_sheet"], request.header_row).head(3)
            result["sample_rows"] = json.loads(sample.to_json(orient="records", date_format="iso"))
            if csv_sheet:
                result["note"] = "CSV has no sheets; supplied sheet was ignored. Use sheet=null for this document."
            return result
        result = query_table_sql(document, sql=request.sql, sheet=selected_sheet,
                                 header_row=request.header_row)
        if csv_sheet:
            result["note"] = "CSV has no sheets; supplied sheet was ignored."
        if result["rows"]:
            signature = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:20]
            item = Evidence(f"sql:{document.id}:{signature}", json.dumps(result, ensure_ascii=False, default=str),
                            Path(document.path).name,
                            f"{('sheet ' + selected_sheet) if selected_sheet else 'table'} · SQL result",
                            document.source_url, 1.0, 4.0)
            existing = next((index for index, source in enumerate(evidence)
                             if source.chunk_id == item.chunk_id), None)
            if existing is None:
                evidence.append(item)
                existing = len(evidence) - 1
            result["ref"] = existing + 1
            result["citation"] = f"[{existing + 1}]"
            result["source"] = item.filename
            result["location"] = item.locator
        return result
