#!/usr/bin/env python
"""Графики сравнения RAG-пайплайна на CPU и на iGPU (llama.cpp Vulkan).

Читает артефакты замеров из ``results/``:
* ``perf_cpu.json`` / ``perf_gpu.json``   — латентность и ток/с (perf-тесты);
* ``load_cpu.json`` / ``load_gpu.json``   — нагрузочные перцентили, RPS, ток/с;
* ``ragas_e5_rerank_bonsai.json``          — RAGAS на CPU (e5-small + реранкер);
* ``comparison_ragas.json``                — RAGAS на iGPU (конфиг ``e5_rerank_igpu``).

Пишет сводный график ``results/compare_cpu_gpu.png`` (2×3 панели).

Примеры:
    uv run python scripts/compare_plots.py
    uv run python scripts/compare_plots.py --out results/compare_cpu_gpu.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from normdoc.config import RESULTS_DIR

try:  # matplotlib — опциональная зависимость
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MPL = True
except Exception:  # pragma: no cover - окружение без matplotlib
    HAS_MPL = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--out", default=str(RESULTS_DIR / "compare_cpu_gpu.png"))
    return p.parse_args()


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def gather(results: Path) -> dict:
    """Собирает сопоставимые метрики CPU и iGPU из артефактов."""
    comparison = _read(results / "comparison_ragas.json")
    gpu_cfg = comparison.get("configs", {}).get("e5_rerank_igpu", {})
    return {
        "perf_cpu": _read(results / "perf_cpu.json"),
        "perf_gpu": _read(results / "perf_gpu.json"),
        "load_cpu": _read(results / "load_cpu.json"),
        "load_gpu": _read(results / "load_gpu.json"),
        "ragas_cpu": _read(results / "ragas_e5_rerank_bonsai.json"),
        "ragas_gpu": {
            "metrics": gpu_cfg.get("metrics", {}),
            "latency_s_mean": gpu_cfg.get("latency_s_mean"),
        },
    }


def _grouped(ax, labels, cpu_vals, gpu_vals, *, title, ylabel, fmt="{:.1f}") -> None:
    x = np.arange(len(labels))
    width = 0.38
    cpu_vals = [v if isinstance(v, (int, float)) else 0 for v in cpu_vals]
    gpu_vals = [v if isinstance(v, (int, float)) else 0 for v in gpu_vals]
    b1 = ax.bar(x - width / 2, cpu_vals, width, label="CPU", color="#9e9e9e")
    b2 = ax.bar(x + width / 2, gpu_vals, width, label="iGPU (Vulkan)", color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h:
                ax.text(bar.get_x() + bar.get_width() / 2, h, fmt.format(h),
                        ha="center", va="bottom", fontsize=7)


def build_figure(data: dict, out_png: Path) -> Path | None:
    """Строит сводный график CPU vs iGPU и сохраняет его в ``out_png``."""
    if not HAS_MPL:
        print("[compare] matplotlib недоступен — графики пропущены")
        return None

    pc, pg = data.get("perf_cpu", {}), data.get("perf_gpu", {})
    lc, lg = data.get("load_cpu", {}), data.get("load_gpu", {})
    rc, rg = data.get("ragas_cpu", {}), data.get("ragas_gpu", {})

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # 1) RAG-латентность (perf-тесты)
    _grouped(
        axes[0, 0],
        ["Ретрив p50", "Генерация p50", "Итого p50"],
        [pc.get("retrieval_s", {}).get("p50"), pc.get("generation_s", {}).get("p50"),
         pc.get("total_s", {}).get("p50")],
        [pg.get("retrieval_s", {}).get("p50"), pg.get("generation_s", {}).get("p50"),
         pg.get("total_s", {}).get("p50")],
        title="RAG: латентность (меньше — лучше)", ylabel="секунды", fmt="{:.1f}",
    )

    # 2) Скорость генерации
    _grouped(
        axes[0, 1],
        ["perf, ток/с", "нагрузка, ток/с"],
        [pc.get("tokens_per_s", {}).get("mean"), lc.get("tokens_per_s", {}).get("mean")],
        [pg.get("tokens_per_s", {}).get("mean"), lg.get("tokens_per_s", {}).get("mean")],
        title="Скорость генерации (больше — лучше)", ylabel="токенов/с", fmt="{:.1f}",
    )

    # 3) RAGAS: качество
    _grouped(
        axes[0, 2],
        ["Faithfulness", "ContextPrecision", "ContextRecall"],
        [rc.get("metrics", {}).get("faithfulness"),
         rc.get("metrics", {}).get("llm_context_precision_with_reference"),
         rc.get("metrics", {}).get("context_recall")],
        [rg.get("metrics", {}).get("faithfulness"),
         rg.get("metrics", {}).get("llm_context_precision_with_reference"),
         rg.get("metrics", {}).get("context_recall")],
        title="RAGAS: качество (e5-small + реранкер)", ylabel="балл (0…1)", fmt="{:.3f}",
    )

    # 4) Нагрузка: перцентили латентности
    _grouped(
        axes[1, 0],
        ["p50", "p95", "p99"],
        [lc.get("latency_s", {}).get("p50"), lc.get("latency_s", {}).get("p95"),
         lc.get("latency_p99_s")],
        [lg.get("latency_s", {}).get("p50"), lg.get("latency_s", {}).get("p95"),
         lg.get("latency_p99_s")],
        title="Нагрузка: перцентили латентности (меньше — лучше)", ylabel="секунды", fmt="{:.1f}",
    )

    # 5) Пропускная способность
    _grouped(
        axes[1, 1],
        ["RPS"],
        [lc.get("rps")],
        [lg.get("rps")],
        title="Нагрузка: пропускная способность (больше — лучше)", ylabel="запросов/с", fmt="{:.3f}",
    )

    # 6) RAGAS: средняя латентность
    _grouped(
        axes[1, 2],
        ["средняя латентность"],
        [rc.get("latency_s_mean")],
        [rg.get("latency_s_mean")],
        title="RAGAS: средняя латентность (меньше — лучше)", ylabel="секунды", fmt="{:.1f}",
    )

    fig.suptitle(
        "Сравнение RAG на CPU и на iGPU (AMD Radeon 780M, llama.cpp Vulkan)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[compare] график: {out_png}")
    return out_png


def main() -> int:
    args = parse_args()
    data = gather(Path(args.results))
    png = build_figure(data, Path(args.out))
    return 0 if png else 1


if __name__ == "__main__":
    raise SystemExit(main())
