"""Local file parsing with locations that can be shown as citations."""

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument
from openpyxl import load_workbook

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".docx"}


@dataclass(frozen=True)
class SourceUnit:
    locator: str
    text: str


def iter_units(path: Path) -> Iterator[SourceUnit]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    yield SourceUnit(f"page {index}", text)
        return
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            yield SourceUnit("text", text)
        return
    if suffix == ".docx":
        document = DocxDocument(path)
        for index, paragraph in enumerate(document.paragraphs, start=1):
            if paragraph.text.strip():
                yield SourceUnit(f"paragraph {index}", paragraph.text.strip())
        for table_no, table in enumerate(document.tables, start=1):
            for row_no, row in enumerate(table.rows, start=1):
                text = " | ".join(cell.text.strip() for cell in row.cells)
                if text.strip(" |"):
                    yield SourceUnit(f"table {table_no} row {row_no}", text)
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return
            for row_no, row in enumerate(reader, start=2):
                text = "; ".join(f"{key}: {value}" for key, value in row.items() if key and value)
                if text:
                    yield SourceUnit(f"row {row_no}", text)
        return
    if suffix == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook:
                rows = sheet.iter_rows(values_only=True)
                headers = next(rows, None)
                if headers is None:
                    continue
                labels = [str(value) if value is not None else f"column_{i + 1}" for i, value in enumerate(headers)]
                for row_no, row in enumerate(rows, start=2):
                    text = "; ".join(
                        f"{label}: {value}"
                        for label, value in zip(labels, row, strict=False)
                        if value is not None
                    )
                    if text:
                        yield SourceUnit(f"sheet {sheet.title} row {row_no}", text)
        finally:
            workbook.close()
        return
    raise ValueError(f"Unsupported file extension: {suffix}")


def split_unit(unit: SourceUnit, max_words: int = 280, overlap: int = 35) -> Iterator[SourceUnit]:
    words = unit.text.split()
    if not words:
        return
    step = max_words - overlap
    if step <= 0:
        raise ValueError("max_words must be greater than overlap")
    for part_no, start in enumerate(range(0, len(words), step), start=1):
        text = " ".join(words[start : start + max_words])
        yield SourceUnit(f"{unit.locator} part {part_no}", text)
        if start + max_words >= len(words):
            break
