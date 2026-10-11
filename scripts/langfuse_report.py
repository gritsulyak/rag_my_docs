#!/usr/bin/env python
"""Экспорт последних трейсов Langfuse в results/langfuse_traces.json для отчёта.

Собирает по каждому трейсу:
* имя, id, timestamp, длительность (latency), ввод/вывод, теги;
* спаны (retrieve/generate) с их latency из metadata.

Требует поднятого Langfuse (облако или self-hosted, профиль monitoring) и
заполненных LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.

Примеры:
    uv run python scripts/langfuse_report.py --limit 50
    LANGFUSE_HOST=http://localhost:3000 uv run python scripts/langfuse_report.py --name rag_answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from normdoc.config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    RESULTS_DIR,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--name", default="", help="Фильтр по имени трейса (например rag_answer)")
    p.add_argument("--out", default=str(RESULTS_DIR / "langfuse_traces.json"))
    p.add_argument(
        "--local",
        default=str(RESULTS_DIR / "traces_local.jsonl"),
        help="Fallback: JSONL-трейсы, если Langfuse-сервер недоступен",
    )
    return p.parse_args()


def _from_local_jsonl(path: Path, name: str, limit: int) -> dict:
    """Fallback: собирает payload из локального JSONL-стока трейсов (TRACE_JSONL)."""
    if not path.exists():
        return {}
    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if name and rec.get("name") != name:
            continue
        spans = rec.get("spans", [])
        question = rec.get("input") or next(
            (s.get("input") for s in spans if s.get("name") == "retrieve" and s.get("input")), ""
        )
        traces.append(
            {
                "id": rec.get("id"),
                "name": rec.get("name"),
                "timestamp": None,
                "latency_s": rec.get("latency_s"),
                "input": question,
                "output": rec.get("output"),
                "tags": ["local"],
                "spans": [
                    {
                        "name": s.get("name"),
                        "type": s.get("name"),
                        "latency_s": s.get("latency_s"),
                        "chunks": (s.get("metadata") or {}).get("chunks"),
                        "model": (s.get("metadata") or {}).get("model"),
                        "usage": s.get("usage"),
                    }
                    for s in spans
                ],
            }
        )
    traces = traces[-limit:]
    latencies = [t["latency_s"] for t in traces if t.get("latency_s")]
    return {
        "available": True,
        "source": "local-jsonl",
        "host": f"file://{path}",
        "count": len(traces),
        "latency_s_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "traces": traces,
    }


def _span_meta(observation) -> dict:
    meta = getattr(observation, "metadata", None) or {}
    if isinstance(meta, dict):
        return meta
    return {}


def export(limit: int, name: str, local_path: str = "") -> dict:
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        local = _from_local_jsonl(Path(local_path), name, limit) if local_path else {}
        if local:
            return local
        return {"available": False, "reason": "LANGFUSE_PUBLIC_KEY/SECRET_KEY не заданы", "traces": []}
    try:
        from langfuse import Langfuse
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"langfuse не установлен: {exc}", "traces": []}

    try:
        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        listing = client.api.trace.list(limit=limit, name=name or None, order_by="timestamp.desc")
    except Exception as exc:  # noqa: BLE001
        local = _from_local_jsonl(Path(local_path), name, limit) if local_path else {}
        if local:
            local["reason"] = f"Langfuse недоступен ({exc}); использован локальный JSONL"
            return local
        return {"available": False, "reason": f"не удалось подключиться: {exc}", "traces": []}

    traces = []
    for item in getattr(listing, "data", []) or []:
        entry = {
            "id": item.id,
            "name": item.name,
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
            "latency_s": getattr(item, "latency", None),
            "input": (item.input or "")[:500] if isinstance(item.input, str) else item.input,
            "output": (item.output or "")[:500] if isinstance(item.output, str) else item.output,
            "tags": list(item.tags or []),
            "spans": [],
        }
        try:
            detail = client.api.trace.get(item.id)
            for obs in getattr(detail, "observations", []) or []:
                meta = _span_meta(obs)
                entry["spans"].append(
                    {
                        "name": getattr(obs, "name", None),
                        "type": getattr(obs, "type", None),
                        "latency_s": meta.get("latency_s") or getattr(obs, "latency", None),
                        "chunks": meta.get("chunks"),
                        "model": meta.get("model"),
                        "usage": getattr(obs, "usage", None) and {
                            "input": getattr(obs.usage, "input", None),
                            "output": getattr(obs.usage, "output", None),
                            "total": getattr(obs.usage, "total", None),
                        },
                    }
                )
        except Exception as exc:  # noqa: BLE001
            entry["detail_error"] = str(exc)
        traces.append(entry)

    latencies = [t["latency_s"] for t in traces if t.get("latency_s")]
    return {
        "available": True,
        "host": LANGFUSE_HOST,
        "count": len(traces),
        "latency_s_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "traces": traces,
    }


def main() -> int:
    args = parse_args()
    payload = export(args.limit, args.name, args.local)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload.get("available"):
        src = payload.get("source", "langfuse")
        print(f"[langfuse] трейсов: {payload['count']} ({src}) -> {args.out}")
    else:
        print(f"[langfuse] недоступен: {payload.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
