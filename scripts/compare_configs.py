#!/usr/bin/env python
"""Сравнение альтернативных конфигураций RAG (LLM × эмбеддинги × реранкер × чанкинг).

Для каждой конфигурации:
  1) при необходимости собирается векторная коллекция (с замером времени);
  2) прогоняется scripts/evaluate_ragas.py на golden-наборе;
  3) результаты собираются в единую таблицу (RAGAS + производительность).

Выход:
  results/comparison_ragas.json — сырые метрики по конфигурациям;
  results/comparison_ragas.md   — готовая markdown-таблица для отчёта.

Примеры:
    uv run python scripts/compare_configs.py --limit 8
    uv run python scripts/compare_configs.py --only baseline,e5_rerank_meno
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from normdoc.config import PROJECT_ROOT, RESULTS_DIR

PY = sys.executable


# Конфигурации: имя -> (overrides окружения, описание)
CONFIGS: dict[str, dict] = {
    "baseline": {
        "desc": "e5-small, без реранкера, bonsai (быстрый бейзлайн)",
        "env": {
            "EMBEDDING_PROFILE": "e5_small",
            "LLM_PROFILE": "bonsai",
            "RERANK_ENABLED": "0",
            "QDRANT_COLLECTION": "normdoc_e5_small",
        },
    },
    "e5_rerank_bonsai": {
        "desc": "e5-small + bge-reranker-v2-m3, bonsai",
        "env": {
            "EMBEDDING_PROFILE": "e5_small",
            "LLM_PROFILE": "bonsai",
            "RERANK_ENABLED": "1",
            "QDRANT_COLLECTION": "normdoc_e5_small",
        },
    },
    "bge_rerank_bonsai": {
        "desc": "BGE-M3 + bge-reranker-v2-m3, bonsai",
        "env": {
            "EMBEDDING_PROFILE": "bge_m3",
            "LLM_PROFILE": "bonsai",
            "RERANK_ENABLED": "1",
            "QDRANT_COLLECTION": "normdoc_bge_m3",
        },
    },
    "e5_rerank_meno": {
        "desc": "e5-small + bge-reranker-v2-m3, Meno-Lite-0.1",
        "env": {
            "EMBEDDING_PROFILE": "e5_small",
            "LLM_PROFILE": "meno",
            "RERANK_ENABLED": "1",
            "QDRANT_COLLECTION": "normdoc_e5_small",
        },
    },
    "bge_rerank_meno": {
        "desc": "BGE-M3 + bge-reranker-v2-m3, Meno-Lite-0.1",
        "env": {
            "EMBEDDING_PROFILE": "bge_m3",
            "LLM_PROFILE": "meno",
            "RERANK_ENABLED": "1",
            "QDRANT_COLLECTION": "normdoc_bge_m3",
        },
    },
    "chars_baseline": {
        "desc": "Символьный чанкинг (600/60) вместо семантического, e5 + rerank, bonsai",
        "env": {
            "EMBEDDING_PROFILE": "e5_small",
            "LLM_PROFILE": "bonsai",
            "RERANK_ENABLED": "1",
            "CHUNKING_MODE": "chars",
            "QDRANT_COLLECTION": "normdoc_e5_chars",
        },
    },
}

# Коллекции, которые нужно собрать под конкретные конфигурации.
COLLECTIONS: dict[str, dict] = {
    "normdoc_e5_small": {"EMBEDDING_PROFILE": "e5_small", "CHUNKING_MODE": "semantic"},
    "normdoc_bge_m3": {"EMBEDDING_PROFILE": "bge_m3", "CHUNKING_MODE": "semantic"},
    "normdoc_e5_chars": {"EMBEDDING_PROFILE": "e5_small", "CHUNKING_MODE": "chars"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="", help="Список имён конфигураций через запятую")
    p.add_argument("--limit", type=int, default=0, help="Вопросов на конфигурацию (0 = все)")
    p.add_argument("--skip-index", action="store_true", help="Не пересобирать коллекции")
    p.add_argument("--gate", action="store_true", help="Прокинуть quality gate в evaluate_ragas.py")
    p.add_argument("--with-perf", action="store_true", help="Дополнительно прогнать perf-тесты (benchmark_rag.py)")
    p.add_argument("--perf-repeat", type=int, default=1, help="Повторов на вопрос в perf-тестах")
    return p.parse_args()


def run(cmd: list[str], env: dict) -> tuple[int, str]:
    full_env = {**os.environ, **env}
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=full_env, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout + proc.stderr


def collection_exists(name: str) -> bool:
    try:
        import requests

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        r = requests.get(f"{url}/collections/{name}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_collections(env_overrides: dict, results: dict) -> None:
    for coll, base_env in COLLECTIONS.items():
        if not any(c["env"].get("QDRANT_COLLECTION") == coll for c in CONFIGS.values()):
            continue
        if collection_exists(coll):
            results.setdefault("index_build_s", {})[coll] = results["index_build_s"].get(coll)
            continue
        print(f"[compare] собираю коллекцию {coll} ({base_env})...")
        env = {k: str(v) for k, v in {**env_overrides, **base_env, "RERANK_ENABLED": "0"}.items()}
        started = time.monotonic()
        code, out = run([PY, "scripts/build_index.py", "--reset"], env)
        elapsed = time.monotonic() - started
        if code != 0:
            print(out[-1500:])
            raise SystemExit(f"Не удалось собрать коллекцию {coll}")
        results.setdefault("index_build_s", {})[coll] = round(elapsed, 1)
        print(f"[compare] {coll}: {elapsed:.1f} c")


def main() -> int:
    args = parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CONFIGS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {"configs": {}, "index_build_s": {}, "n_questions_limit": args.limit}

    if not args.skip_index:
        ensure_collections({}, results)

    for name in names:
        cfg = CONFIGS[name]
        out_path = RESULTS_DIR / f"ragas_{name}.json"
        print(f"\n[compare] === {name}: {cfg['desc']} ===")
        env = {k: str(v) for k, v in cfg["env"].items()}
        # Ограничиваем длину ответа, чтобы bonsai не генерировал по 2 минуты.
        env.setdefault("LLM_MAX_TOKENS", os.getenv("COMPARE_LLM_MAX_TOKENS", "600"))
        if cfg["env"].get("CHUNKING_MODE") == "chars":
            env.setdefault("CHUNK_SIZE", "600")
            env.setdefault("CHUNK_OVERLAP", "60")
        started = time.monotonic()
        ragas_cmd = [
            PY, "scripts/evaluate_ragas.py", "--out", str(out_path), "--limit", str(args.limit)
        ]
        if args.gate:
            ragas_cmd.append("--gate")
        code, out = run(ragas_cmd, env)
        wall = time.monotonic() - started
        if code != 0:
            print(out[-2000:])
            print(f"[compare] ОШИБКА в конфигурации {name}")
            continue
        data = json.loads(out_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        results["configs"][name] = {
            "description": cfg["desc"],
            "collection": cfg["env"]["QDRANT_COLLECTION"],
            "llm": cfg["env"]["LLM_PROFILE"],
            "embeddings": cfg["env"]["EMBEDDING_PROFILE"],
            "rerank": cfg["env"]["RERANK_ENABLED"] == "1",
            "chunking": cfg["env"].get("CHUNKING_MODE", "semantic"),
            "metrics": metrics,
            "latency_s_mean": data.get("latency_s_mean"),
            "retrieval_s_mean": data.get("retrieval_s_mean"),
            "generation_s_mean": data.get("generation_s_mean"),
            "gate": data.get("gate"),
            "wall_s": round(wall, 1),
        }
        print(f"[compare] {name}: {metrics}")

        if args.with_perf:
            perf_path = RESULTS_DIR / f"perf_{name}.json"
            perf_env = {**env, "LLM_MAX_TOKENS": os.getenv("COMPARE_LLM_MAX_TOKENS", "600")}
            pcode, pout = run(
                [
                    PY, "scripts/benchmark_rag.py",
                    "--out", str(perf_path),
                    "--limit", str(args.limit or 6),
                    "--repeat", str(args.perf_repeat),
                    "--name", name,
                ],
                perf_env,
            )
            if pcode != 0:
                print(pout[-1500:])
                print(f"[compare] perf-тест {name} не удался")
            else:
                results["configs"][name]["perf"] = _perf_summary(perf_path)

    _write_outputs(results)
    return 0


def _perf_summary(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "retrieval_s": data.get("retrieval_s"),
        "generation_s": data.get("generation_s"),
        "total_s": data.get("total_s"),
        "tokens_per_s": data.get("tokens_per_s"),
        "answer_chars": data.get("answer_chars"),
    }


def _write_outputs(results: dict) -> None:
    json_path = RESULTS_DIR / "comparison_ragas.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_keys = ["faithfulness", "llm_context_precision_with_reference", "context_recall"]
    header = "| Конфигурация | LLM | Эмбеддинги | Реранкер | Чанкинг | Faithfulness | ContextPrecision | ContextRecall | Ср. латентность, с |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for name, cfg in results["configs"].items():
        m = cfg["metrics"]
        vals = " | ".join(
            f"{m.get(k):.3f}" if isinstance(m.get(k), (int, float)) else "—" for k in metric_keys
        )
        lines.append(
            f"| `{name}` | {cfg['llm']} | {cfg['embeddings']} | {'да' if cfg['rerank'] else 'нет'} "
            f"| {cfg['chunking']} | {vals} | {cfg['latency_s_mean']} |"
        )
    md = "# Сравнение конфигураций RAG\n\n" + "\n".join(lines) + "\n"
    if results.get("index_build_s"):
        md += "\n## Время сборки коллекций\n\n"
        for coll, secs in results["index_build_s"].items():
            md += f"- `{coll}`: {secs} c\n"
    (RESULTS_DIR / "comparison_ragas.md").write_text(md, encoding="utf-8")
    print(f"\n[compare] таблица: {RESULTS_DIR / 'comparison_ragas.md'}")
    print(md)


if __name__ == "__main__":
    raise SystemExit(main())
