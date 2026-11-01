FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
WORKDIR /opt/backend

RUN pip install --no-cache-dir uv==0.11.16
COPY backend/pyproject.toml backend/uv.lock ./
ARG WITH_MODELS=0
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_MODELS" = "1" ]; then uv sync --frozen --no-dev --extra models --no-install-project; \
    else uv sync --frozen --no-dev --no-install-project; fi
COPY backend/README.md ./
COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_MODELS" = "1" ]; then uv sync --frozen --no-dev --extra models; \
    else uv sync --frozen --no-dev; fi
ENV PATH="/opt/backend/.venv/bin:$PATH"

CMD ["uvicorn", "rag_api.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
