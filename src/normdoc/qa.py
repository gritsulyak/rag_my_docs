"""Golden-набор вопросов и статистические помощники для оценок.

Golden-набор (``data/qa/golden_qa.jsonl``) — это строки JSON вида::

    {"id": "q01", "question": "...", "ground_truth": "..."}

Он используется и RAGAS-оценкой (эталонные ответы), и perf-тестами
(набор запросов для замера латентности).
"""

from __future__ import annotations

import json
from pathlib import Path

from normdoc.config import QA_DIR

GOLDEN_QA_PATH = QA_DIR / "golden_qa.jsonl"


def load_qa(path: str | Path = GOLDEN_QA_PATH) -> list[dict]:
    """Читает JSONL-набор вопросов; пустые строки игнорируются."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float | None:
    """Линейная интерполяция перцентиля (q в долях 0..1)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def describe(values: list[float], digits: int = 3) -> dict:
    """mean/p50/p95/min/max по списку значений (round до digits)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    r = lambda x: None if x is None else round(float(x), digits)  # noqa: E731
    return {
        "n": len(clean),
        "mean": r(sum(clean) / len(clean)),
        "p50": r(percentile(clean, 0.50)),
        "p95": r(percentile(clean, 0.95)),
        "min": r(min(clean)),
        "max": r(max(clean)),
    }
