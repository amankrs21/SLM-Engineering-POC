"""Internal CPU embedding and reranking service."""

from functools import lru_cache
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel, Field

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QUERY_TASK = "Given a question about a document collection, retrieve relevant passages that answer it"
embedding_lock = Lock()
rerank_lock = Lock()
app = FastAPI(title="RAG retrieval models")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    is_query: bool = False


class RerankRequest(BaseModel):
    query: str
    texts: list[str] = Field(min_length=1, max_length=50)


@lru_cache
def embedding_model():
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu", model_kwargs={"dtype": torch.float32})


@lru_cache
def reranking_model():
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, device="cpu", model_kwargs={"dtype": torch.float32}, max_length=512)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "embedding_loaded": embedding_model.cache_info().currsize > 0,
            "reranker_loaded": reranking_model.cache_info().currsize > 0}


@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    texts = [f"Instruct: {QUERY_TASK}\nQuery:{text}" for text in request.texts] if request.is_query else request.texts
    with embedding_lock:
        vectors = embedding_model().encode(texts, batch_size=8, normalize_embeddings=True)
    return {"vectors": vectors.tolist(), "model": EMBEDDING_MODEL}


@app.post("/rerank")
def rerank(request: RerankRequest) -> dict:
    pairs = [(request.query, text[:5000]) for text in request.texts]
    with rerank_lock:
        scores = reranking_model().predict(pairs, batch_size=4)
    return {"scores": [float(score) for score in scores], "model": RERANK_MODEL}
