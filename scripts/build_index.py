#!/usr/bin/env python
"""Сборка векторного индекса: PDF -> чанки с метаданными -> Qdrant.

Пример:
    uv run python scripts/build_index.py --reset
    uv run python scripts/build_index.py --chunk-size 400 --chunk-overlap 50

Чанки всегда дублируются в data/chunks/chunks.json — это артефакт для отчёта
и для прогонов RAGAS без повторного парсинга PDF.
"""

from __future__ import annotations

import argparse
import json
import sys

from normdoc.config import CHUNKING_MODE, CHUNKS_JSON, COLLECTION_NAME, CHUNK_OVERLAP, CHUNK_SIZE
from normdoc.indexing import build_index
from normdoc.parsing import Chunk, load_all_chunks, save_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Имя коллекции Qdrant")
    parser.add_argument("--mode", default=CHUNKING_MODE, choices=["semantic", "chars"])
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Только для mode=chars")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Только для mode=chars")
    parser.add_argument("--reset", action="store_true", help="Удалить и пересобрать коллекцию")
    parser.add_argument("--from-cache", action="store_true", help="Взять чанки из data/chunks/chunks.json")
    return parser.parse_args()


def load_cached() -> list[Chunk]:
    payload = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    return [Chunk(text=item["text"], metadata=item["metadata"]) for item in payload]


def main() -> int:
    args = parse_args()

    if args.from_cache and CHUNKS_JSON.exists():
        chunks = load_cached()
        print(f"[build_index] чанков из кэша: {len(chunks)}")
    else:
        chunks = load_all_chunks(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, mode=args.mode
        )
        save_chunks(chunks)
        print(f"[build_index] режим чанкинга: {args.mode}; чанков: {len(chunks)} -> {CHUNKS_JSON}")

    if not chunks:
        print("[build_index] нет чанков: сначала запустите scripts/prepare_docs.py", file=sys.stderr)
        return 1

    build_index(chunks, reset=args.reset, collection_name=args.collection)
    print(f"[build_index] готово: коллекция {args.collection!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
