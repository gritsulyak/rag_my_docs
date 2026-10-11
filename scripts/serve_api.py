#!/usr/bin/env python
"""Запуск HTTP-сервиса RAG-ассистента (FastAPI + uvicorn).

Примеры:
    uv run python scripts/serve_api.py                      # 0.0.0.0:8000
    uv run python scripts/serve_api.py --host 127.0.0.1 --port 8000 --reload
    LLM_PROFILE=meno EMBEDDING_PROFILE=e5_small uv run python scripts/serve_api.py
"""

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Авто-перезапуск при изменении кода")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.reload:
        uvicorn.run("normdoc.api:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run("normdoc.api:app", host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
