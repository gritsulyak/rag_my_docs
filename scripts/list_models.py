#!/usr/bin/env python
"""Печать каталога рекомендованных РФ-моделей для локального RAG.

Примеры:
    uv run python scripts/list_models.py
    uv run python scripts/list_models.py --kind llm
    uv run python scripts/list_models.py --kind embedding --format markdown
"""

from __future__ import annotations

import argparse
import json

from normdoc.model_catalog import CATALOG, ModelInfo, by_kind


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["llm", "embedding", "reranker"], help="Фильтр по типу модели")
    p.add_argument("--format", choices=["table", "markdown", "json"], default="table")
    return p.parse_args()


def _params(m: ModelInfo) -> str:
    if m.active_b:
        return f"{m.params_b}B (активных {m.active_b}B)"
    return f"{m.params_b}B"


def render_table(models: list[ModelInfo]) -> str:
    rows = []
    for m in models:
        rows.append((m.name, m.org, m.kind, _params(m), f"{m.context}", m.hf or "—", m.serve))
    widths = [max(len(str(r[i])) for r in ([("Модель", "Организация", "Тип", "Параметры", "Контекст", "HuggingFace", "Запуск")] + rows)) for i in range(7)]
    def fmt(r):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
    header = ("Модель", "Организация", "Тип", "Параметры", "Контекст", "HuggingFace", "Запуск")
    lines = [fmt(header), fmt(tuple("-" * w for w in widths))]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def render_markdown(models: list[ModelInfo]) -> str:
    lines = [
        "| Модель | Организация | Тип | Параметры | Контекст | Лицензия | HuggingFace | Запуск | Примечание |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        hf = f"[{m.hf}](https://huggingface.co/{m.hf})" if m.hf else "—"
        lines.append(
            f"| {m.name} | {m.org} | {m.kind} | {_params(m)} | {m.context} | {m.license} "
            f"| {hf} | {m.serve} | {m.note} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    models = by_kind(args.kind) if args.kind else CATALOG
    if args.format == "json":
        print(json.dumps([m.as_dict() for m in models], ensure_ascii=False, indent=2))
    elif args.format == "markdown":
        print(render_markdown(models))
    else:
        print(render_table(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
