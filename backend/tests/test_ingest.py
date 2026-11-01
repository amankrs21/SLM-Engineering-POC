from pathlib import Path

import rag_api.ingest as ingest_module
from openpyxl import Workbook
from rag_api.db import db_session
from rag_api.models import Chunk, Corpus, Document
from rag_api.parsers import iter_units
from rag_api.table_tools import inspect_table, query_table
from sqlalchemy import select


class FakeModelService:
    embedding_model_id = "Qwen/Qwen3-Embedding-0.6B"

    def embed(self, texts, *, is_query=False):
        return [[1.0] + [0.0] * 1023 for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.collections = set()
        self.points = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)

    def create_payload_index(self, *args, **kwargs):
        pass

    def upsert(self, collection_name, points, wait):
        self.points.extend(points)


def test_csv_parser_keeps_source_row_number(local_store):
    path = Path(local_store) / "forecasts.csv"
    path.write_text("Region,2026,2027\nSouth Asia,6.3,6.9\n", encoding="utf-8")
    units = list(iter_units(path))
    assert units[0].locator == "row 2"
    assert "South Asia" in units[0].text


def test_folder_ingestion_is_idempotent_and_table_calculation_is_exact(local_store, monkeypatch):
    folder = Path(local_store) / "kb"
    folder.mkdir()
    (folder / "amounts.csv").write_text("item,amount\nA,10.10\nB,0.20\n", encoding="utf-8")
    qdrant = FakeQdrant()
    monkeypatch.setattr(ingest_module, "ModelService", FakeModelService)
    monkeypatch.setattr(ingest_module, "qdrant_client", lambda: qdrant)

    first = ingest_module.ingest_path(folder, "test-v1")
    second = ingest_module.ingest_path(folder, "test-v1")

    assert first == {"indexed": 1, "skipped": 0, "failed": 0}
    assert second == {"indexed": 0, "skipped": 1, "failed": 0}
    assert len(qdrant.points) == 2
    with db_session() as db:
        document = db.scalar(select(Document))
        assert db.get(Corpus, "test-v1").status == "ready"
        assert db.get(Corpus, "test-v1").embedding_model == "Qwen/Qwen3-Embedding-0.6B"
        assert len(db.scalars(select(Chunk)).all()) == 2
        result = query_table(document, sheet=None, filters={}, operation="sum", column="amount")
        assert float(result["result"]) == 10.3


def test_changed_file_requires_new_corpus(local_store, monkeypatch):
    folder = Path(local_store) / "kb"
    folder.mkdir()
    path = folder / "facts.txt"
    path.write_text("first version", encoding="utf-8")
    monkeypatch.setattr(ingest_module, "ModelService", FakeModelService)
    monkeypatch.setattr(ingest_module, "qdrant_client", FakeQdrant)
    assert ingest_module.ingest_path(folder, "test-v1")["indexed"] == 1
    path.write_text("second version", encoding="utf-8")
    assert ingest_module.ingest_path(folder, "test-v1")["failed"] == 1
    with db_session() as db:
        assert db.get(Corpus, "test-v1").status == "ready"


def test_single_uploaded_file_can_be_indexed(local_store, monkeypatch):
    path = Path(local_store) / "upload.txt"
    path.write_text("Uploaded evidence", encoding="utf-8")
    monkeypatch.setattr(ingest_module, "ModelService", FakeModelService)
    qdrant = FakeQdrant()
    monkeypatch.setattr(ingest_module, "qdrant_client", lambda: qdrant)
    assert ingest_module.ingest_path(path, "uploads-v1", display_name="notes.txt") == {
        "indexed": 1, "skipped": 0, "failed": 0}
    assert qdrant.points[0].payload["filename"] == "notes.txt"


def test_excel_sheet_can_be_selected_for_inspection(local_store):
    path = Path(local_store) / "tables.xlsx"
    workbook = Workbook()
    workbook.active.title = "Read Me"
    sheet = workbook.create_sheet("Forecast")
    sheet.append(["Region", "2027"])
    sheet.append(["South Asia", 6.9])
    workbook.save(path)
    document = Document(id="excel", corpus_id="test", path="tables.xlsx", title="Tables",
                        sha256="x" * 64, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    details = inspect_table(document, sheet="Forecast")
    assert details["selected_sheet"] == "Forecast"
    assert details["columns"] == ["Region", "2027"]
    assert details["rows"] == 1


def test_excel_header_row_preserves_physical_row_citation(local_store):
    path = Path(local_store) / "report_table.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Forecast"
    sheet.append(["Report title"])
    sheet.append([])
    sheet.append(["Region", "2026"])
    sheet.append(["World", 2.5])
    workbook.save(path)
    document = Document(id="report-table", corpus_id="test", path="report_table.xlsx", title="Forecast",
                        sha256="x" * 64, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    details = inspect_table(document, sheet="Forecast", header_row=3)
    result = query_table(document, sheet="Forecast", header_row=3,
                         filters={"Region": "World"}, operation="rows")
    assert details["columns"] == ["Region", "2026"]
    assert result["rows"][0]["row"] == 4
    assert result["rows"][0]["values"]["2026"] == "2.5"
