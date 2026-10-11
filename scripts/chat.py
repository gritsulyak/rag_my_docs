#!/usr/bin/env python
"""Интерактивный CLI-чат с ассистентом по нормативным документам.

    uv run python scripts/chat.py
    uv run python scripts/chat.py --question "Что такое вид представления архитектуры?"

Внутри цикла доступны команды:
    :sources  — показать источники последнего ответа,
    :quit     — выход.
"""

from __future__ import annotations

import argparse

from normdoc.assistant import build_assistant
from normdoc.config import COLLECTION_NAME, TOP_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--question", "-q", help="Задать один вопрос и выйти")
    return parser.parse_args()


def print_sources(result) -> None:
    for i, src in enumerate(result.sources, start=1):
        meta = src["metadata"]
        page = meta.get("page")
        location = f", стр. {page}" if page is not None else ""
        score = src.get("score")
        score_str = f" (score={score:.3f})" if isinstance(score, (int, float)) else ""
        print(f"  [{i}] {meta.get('source', '?')}{location}{score_str}")
        snippet = src["text"].strip().replace("\n", " ")
        print(f"      {snippet[:160]}…")


def main() -> int:
    args = parse_args()
    print(f"[chat] коллекция: {args.collection}, top_k={args.top_k}")
    assistant = build_assistant(collection_name=args.collection, top_k=args.top_k)

    if args.question:
        result = assistant.answer(args.question)
        print(f"\n{result.answer}\n")
        print(f"[latency {result.latency_s:.1f} c, usage={result.usage}]")
        print("[sources]")
        print_sources(result)
        return 0

    print("[chat] введите вопрос (пустая строка / :quit — выход, :sources — источники)\n")
    last = None
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question in {":quit", ":q", "exit"}:
            break
        if question == ":sources":
            if last:
                print_sources(last)
            continue

        result = assistant.answer(question)
        print(f"\n{result.answer}\n")
        print(f"[latency {result.latency_s:.1f} c, usage={result.usage}, чанков={len(result.sources)}]")
        if result.dropped_citations:
            print(f"[отброшены выдуманные ссылки: {result.dropped_citations}]")
        print()
        last = result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
