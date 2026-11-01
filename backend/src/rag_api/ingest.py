"""One-shot, idempotent folder ingestion into Postgres and Qdrant."""

import argparse
import fnmatch
import hashlib
import json
import mimetypes
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models as qmodels
from sqlalchemy import select

from rag_api.db import db_session, init_db
from rag_api.models import Chunk, Corpus, Document
from rag_api.parsers import SUPPORTED_EXTENSIONS, iter_units, split_unit
from rag_api.retrieval import ModelService, ensure_collection, qdrant_client
from rag_api.settings import get_settings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_for(root: Path) -> dict[str, dict]:
    path = root / "source_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["path"]: item for item in payload.get("files", [])}


def ingest_file(path: Path, *, root: Path, corpus_id: str, service: ModelService, qdrant,
                display_name: str | None = None) -> str:
    path = path.resolve()
    data_root = Path(get_settings().data_dir).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError(f"File is outside configured data directory: {path}")
    relative = str(path.relative_to(data_root))
    corpus_relative = str(path.relative_to(root))
    metadata = manifest_for(root).get(corpus_relative, {})
    fingerprint = sha256(path)
    with db_session() as db:
        existing = db.scalars(select(Document).where(Document.corpus_id == corpus_id,
                                                       Document.path == relative)).first()
        if existing:
            if existing.sha256 == fingerprint and existing.status == "ready":
                return "skipped"
            if existing.sha256 != fingerprint:
                raise ValueError(f"Changed file requires a new corpus version: {relative}")
        document = existing or Document(
            id=str(uuid5(NAMESPACE_URL, f"{corpus_id}:{relative}:{fingerprint}")),
            corpus_id=corpus_id, path=relative, title=metadata.get("title", display_name or path.name),
            source_url=metadata.get("source_url"), sha256=fingerprint,
            mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream", status="building",
        )
        if not existing:
            db.add(document)
        db.flush()
        chunks = [chunk for unit in iter_units(path) for chunk in split_unit(unit)]
        if not chunks:
            document.status = "failed"
            document.error = "No extractable text; scanned PDFs require OCR"
            return "failed"
        collection = ensure_collection(qdrant, corpus_id)
        for start in range(0, len(chunks), 24):
            batch = chunks[start : start + 24]
            vectors = service.embed([chunk.text for chunk in batch])
            points = []
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk_id = str(uuid5(NAMESPACE_URL, f"{document.id}:{chunk.locator}:{chunk.text}"))
                db.merge(Chunk(id=chunk_id, document_id=document.id, locator=chunk.locator, text=chunk.text))
                points.append(qmodels.PointStruct(id=chunk_id, vector=vector, payload={
                    "corpus_id": corpus_id, "document_id": document.id, "chunk_id": chunk_id,
                    "filename": display_name or path.name, "locator": chunk.locator,
                    "source_url": document.source_url,
                    "derived": bool(metadata.get("derived", False)),
                    "text": chunk.text,
                }))
            qdrant.upsert(collection_name=collection, points=points, wait=True)
            print(f"{relative}: {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")
        document.status = "ready"
        document.error = None
    return "indexed"


def ingest_path(root: Path, corpus_id: str, excludes: list[str] | None = None,
                display_name: str | None = None) -> dict[str, int]:
    init_db()
    root = root.resolve()
    exclusions = excludes or []
    if not root.exists():
        raise FileNotFoundError(root)
    files = [path for path in (root.rglob("*") if root.is_dir() else [root])
             if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
             and not path.name.startswith(".") and not any(fnmatch.fnmatch(path.name, pattern) for pattern in exclusions)]
    if not files:
        raise ValueError(f"No supported files found under {root}")
    service = ModelService()
    with db_session() as db:
        corpus = db.get(Corpus, corpus_id)
        previous_status = corpus.status if corpus is not None else None
        if corpus is None:
            db.add(Corpus(id=corpus_id, status="building", embedding_model=service.embedding_model_id))
        else:
            if corpus.embedding_model != service.embedding_model_id:
                raise ValueError("Corpus embedding model differs from selected provider; use a new corpus ID")
            corpus.status = "building"
    qdrant = qdrant_client()
    summary = {"indexed": 0, "skipped": 0, "failed": 0}
    for path in sorted(files):
        try:
            outcome = ingest_file(path, root=root, corpus_id=corpus_id, service=service,
                                  qdrant=qdrant, display_name=display_name if root.is_file() else None)
            summary[outcome] += 1
            print(f"{outcome}: {path}")
        except Exception as exc:  # noqa: BLE001 - record each parser/provider failure and continue
            summary["failed"] += 1
            print(f"failed: {path}: {exc}")
    with db_session() as db:
        db.get(Corpus, corpus_id).status = (
            "ready" if summary["failed"] == 0 or previous_status == "ready" else "failed"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Index all supported files under a folder")
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--recursive", action="store_true", help="Retained for CLI compatibility; folders recurse")
    parser.add_argument("--exclude", action="append", default=["README.md"])
    args = parser.parse_args()
    result = ingest_path(args.path, args.corpus, args.exclude)
    print(json.dumps(result))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
