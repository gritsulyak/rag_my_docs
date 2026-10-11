"""Опциональный трейсинг запросов в Langfuse.

Совместим с Langfuse SDK v3/v4 (OpenTelemetry-модель: ``start_observation`` /
``start_as_current_observation``); при отсутствии — no-op. Включается при
``LANGFUSE_ENABLED=1`` и заданных ключах.

Каждый RAG-запрос создаёт trace с корневым span'ом ``rag_answer`` и двумя
дочерними наблюдениями:

* ``retrieve`` — поиск чанков (тип ``retriever``);
* ``generate`` — вызов LLM (тип ``generation``, с usage-токенами).

Фиксируются latency, количество найденных чанков и токены (если вернул
llama.cpp/GigaChat). Если Langfuse выключен или ключи пустые — объект работает
как no-op (минимальные накладные расходы), а latency всё равно замеряется и
попадает в текстовый отчёт.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

SpanLike = Any

# Локальный JSONL-сток трейсов (fallback, когда Langfuse-сервер недоступен).
# Включается переменной TRACE_JSONL=/path/to/traces.jsonl; дополняется каждой
# завершённой операцией. Формат совместим с scripts/langfuse_report.py.
_TRACE_JSONL = os.getenv("TRACE_JSONL", "").strip()


class _Timer:
    """Таймер span'а для случая, когда Langfuse выключен."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start


def _usage_details(usage: dict | None) -> dict | None:
    """Нормализует usage llama.cpp/GigaChat к схеме Langfuse (input/output/total)."""
    if not usage:
        return None
    inp = usage.get("prompt_tokens") or usage.get("input_tokens")
    out = usage.get("completion_tokens") or usage.get("output_tokens")
    total = usage.get("total_tokens")
    details = {}
    if inp is not None:
        details["input"] = int(inp)
    if out is not None:
        details["output"] = int(out)
    if total is not None:
        details["total"] = int(total)
    return details or None


class TraceHandle:
    """Контекст трейсинга одного RAG-запроса."""

    def __init__(self, lf=None, query: str = "", name: str = "rag_answer") -> None:
        self._lf = lf
        self._root: SpanLike | None = None
        self._query = query
        self._name = name
        self._timers: dict[str, _Timer] = {}
        self._spans: dict[str, SpanLike] = {}
        self._local_spans: list[dict] = []
        self._active = _Timer(name)
        self._current: _Timer = self._active
        if lf is not None:
            try:
                self._root = lf.start_observation(name=name, as_type="chain", input=query)
            except Exception:  # мониторинг не должен ломать RAG
                self._root = None

    def current_span(self, name: str) -> _Timer:
        """Открывает дочерний span и запускает таймер."""
        timer = _Timer(name)
        self._timers[name] = timer
        self._current = timer
        if self._lf is not None and self._root is not None:
            try:
                as_type = "retriever" if name == "retrieve" else "generation"
                span = self._lf.start_observation(
                    name=name,
                    as_type=as_type,
                    trace_context={
                        "trace_id": self._root.trace_id,
                        "parent_span_id": self._root.id,
                    },
                )
                self._spans[name] = span
            except Exception:
                pass
        return timer

    def record(
        self,
        name: str,
        *,
        input: str = "",
        output: str = "",
        metadata: dict | None = None,
        usage: dict | None = None,
        status: str | None = None,
    ) -> None:
        timer = self._timers.get(name) or self._current
        elapsed = timer.elapsed()
        self._local_spans.append(
            {
                "name": name,
                "latency_s": round(elapsed, 3),
                "input": (input or "")[:500],
                "output": (output or "")[:500],
                "metadata": metadata or {},
                "usage": usage or None,
            }
        )
        span = self._spans.get(name)
        if span is None:
            return
        try:
            span.update(
                input=input or None,
                output=output or None,
                metadata={**(metadata or {}), "latency_s": round(elapsed, 3)},
                usage_details=_usage_details(usage),
                model=metadata.get("model") if metadata else None,
                level="ERROR" if status == "ERROR" else None,
                status_message=status if status not in (None, "OK") else None,
            )
            span.end()
        except Exception:
            pass

    def _write_local(self, output: str, metadata: dict | None, usage: dict | None, status: str | None) -> None:
        """Дописывает трейс в JSONL-файл (если задан TRACE_JSONL)."""
        if not _TRACE_JSONL:
            return
        record = {
            "id": f"local-{int(time.time() * 1000)}",
            "name": self._name,
            "input": self._query,
            "output": (output or "")[:500],
            "latency_s": round(self._active.elapsed(), 3),
            "metadata": metadata or {},
            "usage": usage or None,
            "status": status or "OK",
            "spans": self._local_spans,
        }
        try:
            with open(_TRACE_JSONL, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def finish(
        self,
        output: str,
        metadata: dict | None = None,
        usage: dict | None = None,
        status: str | None = None,
    ) -> None:
        elapsed = self._active.elapsed()
        self._write_local(output, metadata, usage, status)
        if self._root is not None:
            try:
                self._root.update(
                    output=output or None,
                    metadata={**(metadata or {}), "latency_s": round(elapsed, 3)},
                    usage_details=_usage_details(usage),
                    level="ERROR" if status == "ERROR" else None,
                )
                self._root.end()
            except Exception:
                pass
        if self._lf is not None:
            try:
                self._lf.flush()
            except Exception:
                pass

    def result(self) -> dict:
        return {"latency_s": round(self._active.elapsed(), 3)}


class LangfuseTracer:
    """Фабрика трасс; no-op, если мониторинг выключен или SDK недоступен."""

    def __init__(self, *, enabled: bool, public_key: str, secret_key: str, host: str) -> None:
        self.enabled = bool(enabled and public_key and secret_key)
        self._lf = None
        self._host = host
        if self.enabled:
            try:
                from langfuse import Langfuse as _Langfuse

                self._lf = _Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            except Exception as exc:  # pragma: no cover
                print(f"[monitoring] Langfuse недоступен: {exc}")
                self.enabled = False

    def trace(self, name: str, query: str) -> TraceHandle:
        if not self.enabled or self._lf is None:
            return TraceHandle(query=query, name=name)
        return TraceHandle(self._lf, query=query, name=name)


_tracer: LangfuseTracer | None = None


def get_tracer() -> LangfuseTracer:
    global _tracer
    if _tracer is None:
        from normdoc.config import (
            LANGFUSE_ENABLED,
            LANGFUSE_HOST,
            LANGFUSE_PUBLIC_KEY,
            LANGFUSE_SECRET_KEY,
        )

        _tracer = LangfuseTracer(
            enabled=LANGFUSE_ENABLED,
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return _tracer
