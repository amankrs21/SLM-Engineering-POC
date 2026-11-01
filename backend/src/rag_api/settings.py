from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./poc.db"
    qdrant_url: str = "http://localhost:6333"
    retrieval_url: str = "http://localhost:8081"
    slm_base_url: str = "http://localhost:8080/v1"
    slm_model: str = "Qwen3-4B-Q4_K_M"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model: str = "mistral-large-4"
    mistral_api_key: str = ""
    jina_api_key: str = ""
    jina_embedding_model: str = "jina-embeddings-v5-text-small"
    jina_rerank_model: str = "jina-reranker-v3.5"
    cohere_trial_key: str = ""
    cohere_rerank_model: str = "rerank-v4.0-fast"
    embedding_provider: str = "local"
    rerank_provider: str = "local"
    rerank_fallback_provider: str = "none"
    data_dir: str = "./data"
    default_corpus: str = "gep-2026-06-v1"
    phoenix_collector_endpoint: str = ""
    max_upload_mb: int = 25
    top_k: int = 50
    rerank_k: int = 5
    max_output_tokens: int = 512
    mistral_max_output_tokens: int = 1024
    llm_cache_version: str = "agent-v3"
    agent_max_rounds: int = 6
    agent_max_tool_calls: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
