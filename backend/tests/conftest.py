import pytest
from rag_api.db import get_engine, init_db
from rag_api.settings import get_settings


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    (tmp_path / "data").mkdir()
    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()
    yield tmp_path / "data"
    get_engine().dispose()
    get_engine.cache_clear()
    get_settings.cache_clear()
