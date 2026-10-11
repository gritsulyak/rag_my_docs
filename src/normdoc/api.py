"""HTTP-сервис RAG-ассистента (FastAPI).

Эндпоинты:
    GET  /health   — статус, коллекция, модель;
    POST /answer   — вопрос -> ответ, источники, latency, usage (трейс Langfuse);
    GET  /metrics  — метрики Prometheus (latency/throughput для нагрузочных тестов);
    GET  /         — краткая справка.

Запуск локально:
    uv run python scripts/serve_api.py --port 8000
    # или
    uv run uvicorn normdoc.api:app --host 0.0.0.0 --port 8000

В docker-compose сервис ``app`` использует этот же модуль (см. docker/).
"""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from normdoc.assistant import NormDocAssistant, build_assistant
from normdoc.config import COLLECTION_NAME, LLM_MODEL, LLM_PROVIDER, TOP_K

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "rag_requests_total",
    "Всего RAG-запросов",
    labelnames=("status",),
    registry=REGISTRY,
)
TOKENS = Counter(
    "rag_tokens_total",
    "Суммарное число выходных токенов",
    registry=REGISTRY,
)
TOTAL_LATENCY = Histogram(
    "rag_total_latency_seconds",
    "Полная латентность ответа, с",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
    registry=REGISTRY,
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Латентность ретривера, с",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)
GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Латентность генерации LLM, с",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
    registry=REGISTRY,
)
ANSWER_CHARS = Histogram(
    "rag_answer_chars",
    "Длина ответа, символов",
    buckets=(50, 100, 200, 400, 800, 1600, 3200),
    registry=REGISTRY,
)

_ASSISTANTS: dict[tuple[str, int], NormDocAssistant] = {}
_LOCK = threading.Lock()


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос по документам")
    top_k: int | None = Field(None, ge=1, le=20, description="Сколько чанков извлекать")


class Source(BaseModel):
    source: str | None = None
    page: int | None = None
    score: float | None = None
    snippet: str = ""


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float
    retrieval_s: float
    generation_s: float
    usage: dict | None = None
    dropped_citations: list[str] = []
    collection: str
    model: str


app = FastAPI(
    title="normdoc-assistant",
    description="Приватный RAG-ассистент по нормативным документам пользователя со ссылками на источники.",
    version="0.3.0",
)


def get_assistant(collection: str, top_k: int) -> NormDocAssistant:
    key = (collection, top_k)
    with _LOCK:
        if key not in _ASSISTANTS:
            _ASSISTANTS[key] = build_assistant(collection_name=collection, top_k=top_k)
        return _ASSISTANTS[key]


@app.get("/")
def root() -> dict:
    return {
        "service": "normdoc-assistant",
        "collection": COLLECTION_NAME,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "endpoints": ["/health", "/answer", "/metrics", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    top_k = request.top_k or TOP_K
    try:
        assistant = get_assistant(COLLECTION_NAME, top_k)
        result = assistant.answer(request.question)
    except Exception as exc:  # noqa: BLE001
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    REQUESTS.labels(status="ok").inc()
    TOTAL_LATENCY.observe(result.latency_s)
    RETRIEVAL_LATENCY.observe(result.retrieval_s)
    GENERATION_LATENCY.observe(result.generation_s)
    ANSWER_CHARS.observe(len(result.answer))
    completion = (result.usage or {}).get("completion_tokens")
    if completion:
        TOKENS.inc(int(completion))

    sources = [
        Source(
            source=src["metadata"].get("source"),
            page=src["metadata"].get("page"),
            score=src.get("score"),
            snippet=src["text"].strip().replace("\n", " ")[:240],
        )
        for src in result.sources
    ]
    return AnswerResponse(
        answer=result.answer,
        sources=sources,
        latency_s=result.latency_s,
        retrieval_s=result.retrieval_s,
        generation_s=result.generation_s,
        usage=result.usage,
        dropped_citations=result.dropped_citations,
        collection=assistant.collection_name,
        model=LLM_MODEL,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/warmup")
def warmup() -> dict:
    started = time.monotonic()
    assistant = get_assistant(COLLECTION_NAME, TOP_K)
    return {"status": "ready", "collection": assistant.collection_name, "warmup_s": round(time.monotonic() - started, 3)}
