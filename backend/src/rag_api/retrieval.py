"""Qdrant retrieval and HTTP access to the local embedding/reranker service."""

import math
import time
from dataclasses import dataclass

import httpx
from qdrant_client import QdrantClient, models

from rag_api.db import db_session
from rag_api.models import Corpus
from rag_api.observability import observed, record_output
from rag_api.settings import get_settings

VECTOR_SIZE = 1024


def collection_name(corpus_id: str) -> str:
    return "kb_" + "".join(char if char.isalnum() else "_" for char in corpus_id)


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    filename: str
    locator: str
    source_url: str | None
    retrieval_score: float
    rerank_score: float
    derived: bool = False


class ModelService:
    def __init__(self, url: str | None = None, *, embedding_provider: str | None = None):
        settings = get_settings()
        self.url = (url or settings.retrieval_url).rstrip("/")
        self.embedding_provider = embedding_provider or settings.embedding_provider
        if self.embedding_provider not in {"local", "jina"}:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    @property
    def embedding_model_id(self) -> str:
        return (get_settings().jina_embedding_model if self.embedding_provider == "jina"
                else "Qwen/Qwen3-Embedding-0.6B")

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        if not key:
            raise RuntimeError("Hosted retrieval API key is not configured")
        return {"Authorization": f"Bearer {key}"}

    @staticmethod
    def _post_with_rate_retry(client: httpx.Client, url: str, *, headers: dict, json: dict) -> httpx.Response:
        for attempt in range(4):
            response = client.post(url, headers=headers, json=json)
            if getattr(response, "status_code", 200) != 429 or attempt == 3:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After", "60")
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 60
            time.sleep(max(1, min(delay, 75)))
        raise AssertionError("Unreachable retry state")

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        with observed("tool.embedding", "TOOL", {"texts": texts, "is_query": is_query,
                                                   "provider": self.embedding_provider}) as span:
            with httpx.Client(timeout=900) as client:
                if self.embedding_provider == "jina":
                    settings = get_settings()
                    response = self._post_with_rate_retry(
                        client, "https://api.jina.ai/v1/embeddings",
                        headers=self._headers(settings.jina_api_key),
                        json={"model": settings.jina_embedding_model, "input": texts,
                              "task": "retrieval.query" if is_query else "retrieval.passage",
                              "dimensions": VECTOR_SIZE})
                else:
                    response = client.post(f"{self.url}/embed", json={"texts": texts, "is_query": is_query})
                response.raise_for_status()
                payload = response.json()
                vectors = ([item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]
                           if self.embedding_provider == "jina" else payload["vectors"])
            record_output(span, {"model": self.embedding_model_id, "vector_count": len(vectors),
                                 "dimensions": len(vectors[0]) if vectors else 0,
                                 "first_vector_sample": vectors[0][:8] if vectors else []})
        if len(vectors) != len(texts) or any(len(vector) != VECTOR_SIZE for vector in vectors):
            raise ValueError("Embedding service returned unexpected vector dimensions")
        return vectors

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        settings = get_settings()
        provider = settings.rerank_provider
        with observed("tool.rerank", "TOOL", {"query": query, "texts": texts,
                                                "provider": provider}) as span:
            with httpx.Client(timeout=900) as client:
                try:
                    raw_scores, model = self._rerank_with(client, provider, query, texts)
                except httpx.HTTPStatusError as exc:
                    if provider != "jina" or settings.rerank_fallback_provider != "cohere" or not settings.cohere_trial_key:
                        raise
                    span.add_event("Jina rerank failed; using Cohere", {"http.status_code": exc.response.status_code})
                    provider = "cohere"
                    raw_scores, model = self._rerank_with(client, provider, query, texts)
            record_output(span, {"ranked_results": [
                {"input_index": index, "score": float(raw_scores[index])}
                for index in sorted(range(len(raw_scores)), key=lambda item: raw_scores[item], reverse=True)
            ], "model": model, "provider": provider})
            span.set_attribute("rerank.provider", provider)
            scores = ([math.log(max(1e-4, min(1 - 1e-4, float(score))) /
                                (1 - max(1e-4, min(1 - 1e-4, float(score))))) for score in raw_scores]
                      if provider != "local" else raw_scores)
        if len(scores) != len(texts):
            raise ValueError("Reranker returned unexpected score count")
        return [float(score) for score in scores]

    def _rerank_with(self, client: httpx.Client, provider: str, query: str,
                     texts: list[str]) -> tuple[list[float], str]:
        settings = get_settings()
        if provider == "jina":
            model = settings.jina_rerank_model
            response = client.post("https://api.jina.ai/v1/rerank",
                                   headers=self._headers(settings.jina_api_key),
                                   json={"model": model, "query": query, "documents": [text[:5000] for text in texts],
                                         "return_documents": False, "top_n": len(texts)})
        elif provider == "cohere":
            model = settings.cohere_rerank_model
            response = client.post("https://api.cohere.com/v2/rerank",
                                   headers=self._headers(settings.cohere_trial_key),
                                   json={"model": model, "query": query, "documents": [text[:5000] for text in texts],
                                         "top_n": len(texts)})
        elif provider == "local":
            response = client.post(f"{self.url}/rerank", json={"query": query, "texts": texts})
            model = "Qwen/Qwen3-Reranker-0.6B"
        else:
            raise ValueError(f"Unsupported rerank provider: {provider}")
        response.raise_for_status()
        payload = response.json()
        if provider == "local":
            return [float(score) for score in payload["scores"]], payload.get("model", model)
        scores = [0.0] * len(texts)
        seen = set()
        for item in payload["results"]:
            index = item["index"]
            if index in seen or not 0 <= index < len(texts):
                raise ValueError("Hosted reranker returned an invalid candidate index")
            seen.add(index)
            scores[index] = float(item["relevance_score"])
        if len(seen) != len(texts):
            raise ValueError("Hosted reranker omitted candidates")
        return scores, model


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=30)


def ensure_collection(client: QdrantClient, corpus_id: str) -> str:
    name = collection_name(corpus_id)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        for field in ("corpus_id", "document_id"):
            client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    return name


def retrieve(query: str, corpus_id: str, *, top_k: int | None = None, final_k: int | None = None,
             models_service: ModelService | None = None, client: QdrantClient | None = None,
             trace_data: dict | None = None, on_step=None) -> list[Evidence]:
    settings = get_settings()
    if models_service is None:
        with db_session() as db:
            corpus = db.get(Corpus, corpus_id)
            if corpus is None:
                raise ValueError(f"Corpus not found: {corpus_id}")
            provider = "jina" if corpus.embedding_model == settings.jina_embedding_model else "local"
            if provider == "local" and corpus.embedding_model != "Qwen/Qwen3-Embedding-0.6B":
                raise ValueError(f"Unsupported corpus embedding model: {corpus.embedding_model}")
        service = ModelService(embedding_provider=provider)
    else:
        service = models_service
    qdrant = client or qdrant_client()
    if on_step:
        on_step("embedding", "start", {})
    vector = service.embed([query], is_query=True)[0]
    if on_step:
        on_step("embedding", "done", {"dimension": len(vector)})
        on_step("search", "start", {"top_k": top_k or settings.top_k})
    with observed("tool.qdrant_search", "TOOL", {"corpus_id": corpus_id, "query": query,
                                                  "vector_dimensions": len(vector),
                                                  "top_k": top_k or settings.top_k}) as search_span:
        result = qdrant.query_points(
            collection_name=collection_name(corpus_id),
            query=vector,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="corpus_id", match=models.MatchValue(value=corpus_id))]),
            limit=top_k or settings.top_k,
            with_payload=True,
        )
        record_output(search_span, {"candidate_count": len(result.points), "candidates": [
            {"id": str(point.id), "score": float(point.score),
             "source": (point.payload or {}).get("filename"),
             "location": (point.payload or {}).get("locator"),
             "preview": str((point.payload or {}).get("text", ""))[:400]}
            for point in result.points]})
    points = result.points
    if on_step:
        on_step("search", "done", {"candidates": len(points)})
    if trace_data is not None:
        trace_data["candidate_ids"] = [str(point.id) for point in points]
        trace_data["candidate_scores"] = [float(point.score) for point in points]
    if not points:
        return []
    if on_step:
        on_step("rerank", "start", {"candidates": len(points)})
    scores = service.rerank(query, [str((point.payload or {}).get("text", "")) for point in points])
    if on_step:
        on_step("rerank", "done", {"selected": final_k or settings.rerank_k})
    if trace_data is not None:
        trace_data["rerank_scores"] = scores
    # A concise paraphrased note can outrank its original report. Prefer primary
    # evidence when its rerank score is reasonably close (2.5 logit margin).
    ranked = sorted(zip(points, scores, strict=True), key=lambda item: (
        item[1] - 2.5 if (item[0].payload or {}).get("derived") else item[1]
    ), reverse=True)
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for point, score in ranked:
        payload = point.payload or {}
        text = str(payload.get("text", ""))
        fingerprint = " ".join(text.lower().split())[:240]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        evidence.append(Evidence(
            chunk_id=str(payload.get("chunk_id", point.id)), text=text,
            filename=str(payload.get("filename", "unknown")), locator=str(payload.get("locator", "unknown")),
            source_url=payload.get("source_url"), retrieval_score=float(point.score), rerank_score=score,
            derived=bool(payload.get("derived", False)),
        ))
        if len(evidence) >= (final_k or settings.rerank_k):
            break
    return evidence
