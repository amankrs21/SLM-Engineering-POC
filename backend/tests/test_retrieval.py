from types import SimpleNamespace

from rag_api.retrieval import retrieve


class FakeModels:
    def embed(self, texts, *, is_query=False):
        return [[1.0] + [0.0] * 1023]

    def rerank(self, query, texts):
        return [9.0, 8.0]


class FakeQdrant:
    def query_points(self, **kwargs):
        note = SimpleNamespace(id="note", score=0.9, payload={
            "text": "Global growth is 2.5 in 2026", "filename": "summary.txt",
            "locator": "text", "derived": True,
        })
        report = SimpleNamespace(id="report", score=0.8, payload={
            "text": "World Bank report: global GDP growth for 2026 is 2.5 percent",
            "filename": "report.pdf", "locator": "page 1", "derived": False,
        })
        return SimpleNamespace(points=[note, report])


def test_primary_source_preferred_over_similar_derived_note():
    evidence = retrieve("global growth 2026", "test", top_k=2, final_k=2,
                        models_service=FakeModels(), client=FakeQdrant())
    assert [item.filename for item in evidence] == ["report.pdf", "summary.txt"]
    assert evidence[1].derived is True
