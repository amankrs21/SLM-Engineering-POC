"""Knowledge base administration and structured table tools."""

import asyncio
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import func, select

from rag_api.api.schemas import IngestRequest, TableQuery
from rag_api.db import db_session
from rag_api.ingest import ingest_path
from rag_api.models import Corpus, Document
from rag_api.settings import get_settings
from rag_api.table_tools import inspect_table, query_table

router = APIRouter()

@router.post("/v1/corpora/{corpus_id}/ingestions")
async def start_ingestion(corpus_id: str, request: IngestRequest) -> dict:
    root = Path(get_settings().data_dir).resolve()
    folder = (root / request.folder).resolve()
    if not folder.is_relative_to(root):
        raise HTTPException(400, "Folder must be inside the configured data directory")
    try:
        result = await asyncio.to_thread(ingest_path, folder, corpus_id, ["*.zip", "README.md"])
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"corpus_id": corpus_id, **result}


@router.get("/v1/corpora/{corpus_id}")
def corpus_status(corpus_id: str) -> dict:
    with db_session() as db:
        corpus = db.get(Corpus, corpus_id)
        if corpus is None:
            raise HTTPException(404, "Corpus not found")
        counts = db.execute(select(Document.status, func.count(Document.id)).where(
            Document.corpus_id == corpus_id).group_by(Document.status)).all()
        return {"corpus_id": corpus_id, "status": corpus.status,
                "documents": {status: count for status, count in counts}}


@router.post("/v1/uploads")
async def upload_file(file: Annotated[UploadFile, File()], corpus_id: str | None = None) -> dict:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    from rag_api.parsers import SUPPORTED_EXTENSIONS

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "Upload exceeds file size limit")
    folder = Path(settings.data_dir).resolve() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}_{Path(file.filename or 'upload').name}"
    path = folder / filename
    path.write_bytes(content)
    target_corpus = corpus_id or settings.default_corpus
    result = await asyncio.to_thread(ingest_path, path, target_corpus, [], Path(file.filename or "upload").name)
    if result["failed"]:
        raise HTTPException(422, "Uploaded file could not be indexed; check its format and content")
    return {"filename": filename, "corpus_id": target_corpus, "ingestion": result}


@router.get("/v1/documents")
def list_documents(corpus_id: str | None = None) -> list[dict]:
    target = corpus_id or get_settings().default_corpus
    with db_session() as db:
        documents = db.scalars(select(Document).where(Document.corpus_id == target).order_by(Document.path)).all()
        return [{"id": item.id, "title": item.title, "path": item.path, "status": item.status,
                 "source_url": item.source_url} for item in documents]


@router.get("/v1/tables/{document_id}")
def table_details(document_id: str, sheet: str | None = None, header_row: int = 1) -> dict:
    with db_session() as db:
        document = db.get(Document, document_id)
        if document is None:
            raise HTTPException(404, "Document not found")
        try:
            return inspect_table(document, sheet, header_row)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


@router.post("/v1/tables/query")
def table_query(request: TableQuery) -> dict:
    with db_session() as db:
        document = db.get(Document, request.document_id)
        if document is None:
            raise HTTPException(404, "Document not found")
        try:
            return query_table(document, sheet=request.sheet, filters=request.filters,
                               operation=request.operation, column=request.column,
                               header_row=request.header_row)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
