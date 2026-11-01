"""Persistent exact-prompt LLM response cache without storing prompt text."""

import hashlib
import json

from sqlalchemy.exc import IntegrityError

from rag_api.db import db_session
from rag_api.models import LLMCache
from rag_api.settings import get_settings


def generation_cache_key(branch: str, model: str, corpus_id: str, messages: list[dict], *,
                         tools: list[dict] | None = None, tool_choice: str = "auto") -> str:
    settings = get_settings()
    payload = {
        "version": settings.llm_cache_version,
        "branch": branch,
        "model": model,
        "corpus_id": corpus_id,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "low" if branch == "mistral" else None,
        "max_tokens": settings.mistral_max_output_tokens if branch == "mistral" else settings.max_output_tokens,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached(key: str) -> LLMCache | None:
    with db_session() as db:
        return db.get(LLMCache, key)


def store_cached(key: str, *, conversation_id: str, corpus_id: str, branch: str,
                 model: str, answer: str, prompt_tokens: int | None,
                 completion_tokens: int | None) -> None:
    try:
        with db_session() as db:
            if db.get(LLMCache, key) is None:
                db.add(LLMCache(key=key, source_conversation_id=conversation_id,
                                corpus_id=corpus_id, branch=branch, model=model,
                                answer=answer, prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens))
    except IntegrityError:
        # Concurrent identical first turns may write the same cache key.
        pass
