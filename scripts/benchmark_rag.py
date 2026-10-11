#!/usr/bin/env python
"""Perf-тесты RAG-пайплайна: скорость ретривера, генерации и токенов/с.

Прогоняет golden-набор вопросов через ассистент и замеряет:
* ``retrieval_s``   — латентность поиска (эмбеддинг запроса + Qdrant + реранкер);
* ``generation_s``  — латентность генерации LLM;
* ``total_s``       — суммарная латентность ответа;
* ``tokens_out``    — токенов в ответе (из usage llama.cpp/GigaChat API);
* ``tokens_per_s``  — скорость генерации, ток/с;
* ``answer_chars``  — длина ответа в символах;
* ``chunks``        — сколько контекстов извлечено.

Примеры:
    uv run python scripts/benchmark_rag.py --limit 6 --repeat 2
    LLM_PROFILE=meno EMBEDDING_PROFILE=e5_small \
        uv run python scripts/benchmark_rag.py --out results/perf_meno.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from normdoc.assistant import build_assistant
from normdoc.config import (
    COLLECTION_NAME,
    LLM_MODEL,
    LLM_PROVIDER,
    PROJECT_ROOT,
    RERANK_ENABLED,
    RESULTS_DIR,
    TOP_K,
)
from normdoc.qa import GOLDEN_QA_PATH, describe, load_qa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qa", default=str(GOLDEN_QA_PATH))
    p.add_argument("--collection", default=COLLECTION_NAME)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--limit", type=int, default=0, help="Ограничить число вопросов (0 = все)")
    p.add_argument("--repeat", type=int, default=1, help="Прогонов на вопрос (для стабильности)")
    p.add_argument("--warmup", type=int, default=1, help="Прогревочных запросов (загрузка моделей)")
    p.add_argument("--name", default="", help="Имя конфигурации для имени файла")
    p.add_argument("--out", default="", help="Куда сохранить JSON (по умолчанию results/perf_<name>.json)")
    return p.parse_args()


def _completion_tokens(usage: dict) -> float | None:
    value = usage.get("completion_tokens") or usage.get("output_tokens")
    return float(value) if value else None


def main() -> int:
    args = parse_args()
    rows = load_qa(args.qa)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"Пустой golden-набор: {args.qa}")

    name = args.name or f"{LLM_PROVIDER}_{LLM_MODEL}_{args.collection}".replace("/", "_")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"perf_{name}.json"

    print(f"[perf] конфигурация={name}, коллекция={args.collection}, вопросов={len(rows)}, repeat={args.repeat}")
    assistant = build_assistant(collection_name=args.collection, top_k=args.top_k)

    for i in range(args.warmup):
        warm = assistant.answer(rows[0]["question"])
        print(f"[perf] прогрев {i + 1}/{args.warmup}: {warm.latency_s:.1f} c")

    retrieval, generation, total, tokens_out, tps, chars, chunks = ([] for _ in range(7))
    per_run: list[dict] = []
    for row in rows:
        for rep in range(args.repeat):
            result = assistant.answer(row["question"])
            usage = result.usage or {}
            comp = _completion_tokens(usage)
            retrieval.append(result.retrieval_s)
            generation.append(result.generation_s)
            total.append(result.latency_s)
            chars.append(len(result.answer))
            chunks.append(len(result.sources))
            if comp is not None:
                tokens_out.append(comp)
                if result.generation_s > 0:
                    tps.append(comp / result.generation_s)
            per_run.append(
                {
                    "id": row.get("id", "?"),
                    "repeat": rep,
                    "retrieval_s": result.retrieval_s,
                    "generation_s": result.generation_s,
                    "total_s": result.latency_s,
                    "tokens_out": comp,
                    "chunks": len(result.sources),
                }
            )
            print(
                f"[perf] {row.get('id', '?')}#{rep}: retrieve={result.retrieval_s:.2f} c, "
                f"gen={result.generation_s:.2f} c, total={result.latency_s:.2f} c, tokens={comp}"
            )

    payload = {
        "name": name,
        "collection": args.collection,
        "llm_model": LLM_MODEL,
        "llm_provider": LLM_PROVIDER,
        "rerank": RERANK_ENABLED,
        "top_k": args.top_k,
        "n_questions": len(rows),
        "repeat": args.repeat,
        "retrieval_s": describe(retrieval),
        "generation_s": describe(generation),
        "total_s": describe(total),
        "tokens_out": describe(tokens_out),
        "tokens_per_s": describe(tps),
        "answer_chars": describe(chars),
        "chunks": describe(chunks),
        "per_run": per_run,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[perf] итоги:")
    print(f"  retrieval_s  mean={payload['retrieval_s']['mean']} p95={payload['retrieval_s']['p95']}")
    print(f"  generation_s mean={payload['generation_s']['mean']} p95={payload['generation_s']['p95']}")
    print(f"  total_s      mean={payload['total_s']['mean']} p95={payload['total_s']['p95']}")
    print(f"  tokens_per_s mean={payload['tokens_per_s']['mean']}")
    print(f"[perf] сохранено: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
