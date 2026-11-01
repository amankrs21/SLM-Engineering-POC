"""Validated, read-only spreadsheet operations."""

import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from rag_api.models import Document
from rag_api.settings import get_settings

SQL_FUNCTIONS = {"avg", "sum", "total", "count", "min", "max", "round", "abs", "coalesce", "ifnull",
                 "nullif", "lower", "upper", "trim", "ltrim", "rtrim", "length", "substr", "substring",
                 "replace", "like", "glob", "date", "strftime"}


def document_path(document: Document) -> Path:
    root = Path(get_settings().data_dir).resolve()
    path = (root / document.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Document path is missing or outside the data directory")
    return path


def read_table(document: Document, sheet: str | None = None, header_row: int = 1) -> pd.DataFrame:
    if not 1 <= header_row <= 100:
        raise ValueError("Header row must be between 1 and 100")
    path = document_path(document)
    if path.suffix.lower() == ".csv":
        if sheet:
            raise ValueError("CSV files do not have sheets")
        frame = pd.read_csv(path, encoding="utf-8-sig", header=header_row - 1)
    elif path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name=sheet or 0, header=header_row - 1, engine="openpyxl")
    else:
        raise ValueError("Only CSV and XLSX support table queries")
    if len(frame) > 100_000:
        raise ValueError("Table exceeds POC row limit")
    return frame


def inspect_table(document: Document, sheet: str | None = None, header_row: int = 1) -> dict:
    path = document_path(document)
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            sheets = workbook.sheetnames
        finally:
            workbook.close()
    elif path.suffix.lower() == ".csv":
        sheets = []
    else:
        raise ValueError("Only CSV and XLSX support table inspection")
    selected_sheet = sheet or (sheets[0] if sheets else None)
    if sheet and sheet not in sheets:
        raise ValueError(f"Unknown sheet: {sheet}. Available sheets: {sheets}. Use sheet=null for initial inspection.")
    frame = read_table(document, selected_sheet, header_row)
    return {"document_id": document.id, "filename": path.name, "sheets": sheets,
            "selected_sheet": selected_sheet, "header_row": header_row,
            "columns": [str(name) for name in frame.columns], "rows": len(frame)}


def query_table(document: Document, *, sheet: str | None, filters: dict[str, str],
                operation: str, column: str | None = None, header_row: int = 1) -> dict:
    frame = read_table(document, sheet, header_row)
    for name, value in filters.items():
        if name not in frame.columns:
            raise ValueError(f"Unknown column: {name}")
        frame = frame[frame[name].astype(str).str.casefold() == value.casefold()]
    if operation == "rows":
        selected = frame.head(25)
        result = None
    elif operation == "count":
        selected = frame.head(25)
        result = len(frame)
    elif operation in {"sum", "average"}:
        if column not in frame.columns:
            raise ValueError(f"Unknown column: {column}")
        try:
            numbers = [Decimal(str(value)) for value in frame[column].dropna()]
        except InvalidOperation as exc:
            raise ValueError(f"Column is not numeric: {column}") from exc
        total = sum(numbers, Decimal(0))
        result = str(total if operation == "sum" else total / len(numbers)) if numbers else None
        selected = frame.head(25)
    else:
        raise ValueError("Allowed operations: rows, count, sum, average")
    return {"document_id": document.id, "sheet": sheet, "operation": operation, "column": column,
            "filters": filters, "result": result, "matched_rows": len(frame),
            "rows": [{"row": int(index) + header_row + 1,
                      "values": {str(key): str(value) for key, value in row.items() if not pd.isna(value)}}
                     for index, row in selected.iterrows()]}


def query_table_sql(document: Document, *, sql: str, sheet: str | None = None,
                    header_row: int = 1) -> dict:
    """Execute one bounded SELECT against this document, never the application DB."""
    if len(sql) > 4000 or not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Use a single SELECT query against the data table (maximum 4000 characters)")
    frame = read_table(document, sheet, header_row).copy()
    if "_source_row" in frame.columns:
        raise ValueError("Reserved column name: _source_row")
    frame.insert(0, "_source_row", [int(index) + header_row + 1 for index in frame.index])
    with sqlite3.connect(":memory:") as connection:
        frame.to_sql("data", connection, index=False)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        connection.execute("PRAGMA query_only = ON")

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_READ:
                return sqlite3.SQLITE_OK if arg1 == "data" else sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_OK if (arg2 or "").lower() in SQL_FUNCTIONS else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        ticks = 0

        def bounded_execution():
            nonlocal ticks
            ticks += 1
            return ticks > 2000

        connection.set_progress_handler(bounded_execution, 1000)
        try:
            cursor = connection.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(26)
        except sqlite3.Error as exc:
            raise ValueError(f"Read-only SQL query failed: {exc}") from exc
    return {"document_id": document.id, "sheet": sheet, "header_row": header_row, "sql": sql,
            "columns": columns, "rows": [dict(zip(columns, row, strict=True)) for row in rows[:25]],
            "truncated": len(rows) > 25}
