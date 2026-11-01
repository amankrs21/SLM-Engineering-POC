"""Hosted adapters preserve vector order and candidate-to-score mapping."""

import math
from typing import ClassVar

from rag_api import retrieval
from rag_api.settings import get_settings


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeClient:
    requests: ClassVar[list] = []

    def __init__(self, timeout):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def post(self, url, *, headers=None, json):
        self.requests.append((url, json))
        if url.endswith("/embeddings"):
            return FakeResponse({"data": [
                {"index": 1, "embedding": [2.0] * 1024},
                {"index": 0, "embedding": [1.0] * 1024},
            ]})
        return FakeResponse({"results": [
            {"index": 1, "relevance_score": 0.1},
            {"index": 0, "relevance_score": 0.9},
        ]})


def test_jina_embedding_and_rerank_keep_input_order(local_store, monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "jina")
    monkeypatch.setenv("RERANK_PROVIDER", "jina")
    get_settings.cache_clear()
    FakeClient.requests = []
    monkeypatch.setattr(retrieval.httpx, "Client", FakeClient)

    service = retrieval.ModelService()
    vectors = service.embed(["first", "second"], is_query=True)
    scores = service.rerank("question", ["first", "second"])

    assert vectors[0][0] == 1.0 and vectors[1][0] == 2.0
    assert FakeClient.requests[0][1]["task"] == "retrieval.query"
    assert FakeClient.requests[0][1]["dimensions"] == 1024
    assert FakeClient.requests[1][1]["model"] == "jina-reranker-v3.5"
    assert math.isclose(scores[0], math.log(9), rel_tol=1e-6)
    assert math.isclose(scores[1], -math.log(9), rel_tol=1e-6)


def test_jina_embedding_rate_limit_retries(monkeypatch):
    class RateResponse:
        status_code = 429
        headers: ClassVar[dict[str, str]] = {"Retry-After": "2"}

        def raise_for_status(self):
            raise AssertionError("Retry should happen first")

    class RateClient:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return RateResponse() if self.calls == 1 else FakeResponse({"data": []})

    delays = []
    monkeypatch.setattr(retrieval.time, "sleep", delays.append)
    client = RateClient()
    response = retrieval.ModelService._post_with_rate_retry(
        client, "https://api.jina.ai/v1/embeddings", headers={}, json={})
    assert client.calls == 2
    assert delays == [2.0]
    assert response.json() == {"data": []}
