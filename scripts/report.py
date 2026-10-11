#!/usr/bin/env python
"""Сборка единого отчёта results/REPORT.md из артефактов оценок.

Объединяет:
* RAGAS-метрики (``results/comparison_ragas.json``);
* perf-тесты (``results/perf_*.json``);
* нагрузочные тесты и графики (``results/load_*.json`` + ``results/load_*.png``);
* трейсы Langfuse (``results/langfuse_traces.json``, если мониторинг включён);
* результаты pytest (``results/pytest.txt``);
* каталог рекомендованных РФ-моделей;
* docker-compose профили железа;
* сведения о стенде и корпусе.

Примеры:
    uv run python scripts/report.py
    uv run python scripts/report.py --out results/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

from normdoc.config import (
    CHUNKS_JSON,
    COLLECTION_NAME,
    LLM_MODEL,
    LLM_PROVIDER,
    MANIFEST_PATH,
    QDRANT_URL,
    RESULTS_DIR,
)
from normdoc.model_catalog import CATALOG
from normdoc.reporting import build_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(RESULTS_DIR / "REPORT.md"))
    p.add_argument("--comparison", default=str(RESULTS_DIR / "comparison_ragas.json"))
    return p.parse_args()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _total_ram_gb() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
        if match:
            return round(int(match.group(1)) / 1024 / 1024, 1)
    except Exception:
        pass
    return None


def gather_environment() -> dict:
    env = {
        "ОС": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "CPU": f"{platform.processor() or 'CPU'} x{platform.os.cpu_count() if hasattr(platform.os, 'cpu_count') else '?'}",
        "RAM, ГБ": _total_ram_gb(),
        "Провайдер LLM": LLM_PROVIDER,
        "LLM": LLM_MODEL,
        "Коллекция по умолчанию": COLLECTION_NAME,
        "Qdrant": QDRANT_URL,
    }
    import os

    try:
        env["CPU"] = f"{platform.machine()} x{os.cpu_count()}"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            env["GPU"] = torch.cuda.get_device_name(0)
        else:
            env["GPU"] = "нет (CUDA)"
    except Exception:
        env["GPU"] = "torch недоступен"
    # Бэкенд инференса LLM задаётся явно (llama.cpp Vulkan/ROCm/CUDA, vLLM, …):
    # torch не видит iGPU AMD, поэтому без этой переменной стенд выглядел бы «CPU».
    serving = os.getenv("LLM_SERVING", "").strip()
    if serving:
        env["LLM-инференс"] = serving
        if env.get("GPU") in (None, "нет (CUDA)"):
            env["GPU"] = serving
    return env


def gather_corpus() -> dict:
    manifest = _read_json(MANIFEST_PATH)
    files = manifest.get("files", [])
    chunks = _read_json(CHUNKS_JSON)
    kinds: dict[str, int] = {}
    for entry in files:
        kinds[entry.get("kind", "?")] = kinds.get(entry.get("kind", "?"), 0) + 1
    return {
        "Документов": len(files),
        "Чанков": len(chunks) if isinstance(chunks, list) else None,
        "Типы": ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) or None,
        "Дублей исключено": len(manifest.get("deduplicated", [])),
    }


def load_perfs() -> list[dict]:
    perfs: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("perf_*.json")):
        data = _read_json(path)
        if data:
            data.setdefault("name", path.stem.removeprefix("perf_"))
            perfs.append(data)
    return perfs


def load_loads() -> list[dict]:
    """Нагрузочные тесты: JSON + привязка PNG-графика рядом с ним."""
    loads: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("load_*.json")):
        data = _read_json(path)
        if not data:
            continue
        data.setdefault("name", path.stem.removeprefix("load_"))
        png = path.with_suffix(".png")
        if png.exists():
            data["_png"] = str(png)
        loads.append(data)
    return loads


def load_langfuse() -> dict:
    return _read_json(RESULTS_DIR / "langfuse_traces.json")


def load_ragas_single() -> dict:
    """Контрольный одиночный прогон RAGAS на golden-наборе (results/ragas_golden.json)."""
    return _read_json(RESULTS_DIR / "ragas_golden.json")


def gather_compose_info() -> dict:
    """Пытается вытащить список сервисов из docker-compose, если docker доступен."""
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return {}
    try:
        out = subprocess.run(
            ["docker", "compose", "--profile", "ai395", "--profile", "monitoring", "config", "--services"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if out.returncode == 0:
            return {"services": [s.strip() for s in out.stdout.splitlines() if s.strip()]}
    except Exception:
        pass
    return {}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    comparison = _read_json(Path(args.comparison))
    perfs = load_perfs()
    loads = load_loads()
    langfuse = load_langfuse()
    pytest_text = _read_text(RESULTS_DIR / "pytest.txt")

    report = build_report(
        comparison=comparison,
        perfs=perfs,
        pytest_text=pytest_text,
        models=CATALOG,
        environment=gather_environment(),
        corpus=gather_corpus(),
        loads=loads,
        langfuse=langfuse,
        compose_info=gather_compose_info(),
        ragas_single=load_ragas_single(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[report] собран {out}")
    print(f"[report] конфигураций RAGAS: {len(comparison.get('configs', {}))}, perf: {len(perfs)}, "
          f"load: {len(loads)}, langfuse: {'да' if langfuse.get('available') else 'нет'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
