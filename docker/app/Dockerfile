# Образ RAG-приложения (normdoc-assistant): FastAPI + LlamaIndex + Qdrant + Langfuse.
FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    # Эмбеддинги/реранкер считаются на CPU (EMBEDDING_DEVICE=cpu),
    # поэтому тянем CPU-сборку torch — образ в разы меньше и без CUDA-колёс.
    UV_TORCH_BACKEND=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv — быстрый менеджер пакетов (см. pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

WORKDIR /app

# Сначала зависимости (кэшируется при изменении только исходников).
# --no-install-project: на этом слое кода проекта ещё нет, ставим только зависимости.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project --extra perf

# Затем код проекта и вспомогательные каталоги.
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY data/qa ./data/qa
COPY README.md README_RU.md ./

# Теперь ставим сам пакет normdoc (editable) из скопированного src/.
RUN uv sync --no-dev --extra perf

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# По умолчанию — HTTP API. Другие команды задаются в docker-compose (индексация,
# RAGAS, load-тесты) через `command:`.
CMD ["uvicorn", "normdoc.api:app", "--host", "0.0.0.0", "--port", "8000"]
