#!/usr/bin/env python
"""Нагрузочное тестирование инференса: HTTP-нагрузка на /answer + графики.

Замеряет под конкурентностью:
* пропускную способность (RPS);
* перцентили латентности (p50/p90/p95/p99) и распределение;
* долю ошибок;
* токены/с (из usage ответа, если LLM его возвращает).

Результаты: ``results/load_<name>.json`` и графики ``results/load_<name>.png``.
Скрипт бьёт по уже запущенному API (``scripts/serve_api.py``) или по
docker-сервису ``app``.

Примеры:
    uv run python scripts/load_test.py --requests 60 --concurrency 4
    uv run python scripts/load_test.py --url http://localhost:8000 --name docker --concurrency 8
    uv run python scripts/load_test.py --requests 30 --plot-only results/load_default.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from normdoc.config import RESULTS_DIR
from normdoc.qa import GOLDEN_QA_PATH, describe, load_qa, percentile

try:  # matplotlib — опциональная зависимость (группа perf)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Базовый URL API")
    p.add_argument("--qa", default=str(GOLDEN_QA_PATH))
    p.add_argument("--requests", type=int, default=40, help="Всего запросов")
    p.add_argument("--concurrency", type=int, default=4, help="Число параллельных воркеров")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=1, help="Прогревочных запросов")
    p.add_argument("--name", default="default", help="Имя набора результатов")
    p.add_argument("--out", default="", help="JSON (по умолчанию results/load_<name>.json)")
    p.add_argument("--plot-only", default="", help="Только построить графики по существующему JSON")
    return p.parse_args()


def _one_request(url: str, question: str, timeout: float) -> dict:
    import httpx

    started = time.monotonic()
    try:
        resp = httpx.post(f"{url}/answer", json={"question": question}, timeout=timeout)
        elapsed = time.monotonic() - started
        if resp.status_code != 200:
            return {"ok": False, "latency_s": elapsed, "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        return {
            "ok": True,
            "latency_s": elapsed,
            "server_latency_s": data.get("latency_s"),
            "retrieval_s": data.get("retrieval_s"),
            "generation_s": data.get("generation_s"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "chars": len(data.get("answer", "")),
            "sources": len(data.get("sources", [])),
            "question": question,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "latency_s": time.monotonic() - started, "error": str(exc)}


def run_load(args: argparse.Namespace, questions: list[str]) -> dict:
    print(f"[load] прогрев ({args.warmup})...")
    for _ in range(args.warmup):
        _one_request(args.url, questions[0], args.timeout)

    plan = [questions[i % len(questions)] for i in range(args.requests)]
    print(f"[load] {args.requests} запросов, конкурентность {args.concurrency}, URL {args.url}")

    results: list[dict] = []
    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_one_request, args.url, q, args.timeout) for q in plan]
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            if i % 10 == 0 or i == len(futures):
                print(f"[load] готово {i}/{len(futures)}")
    wall = time.monotonic() - wall_start

    latencies = [r["latency_s"] for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    tokens = [r["completion_tokens"] for r in results if r.get("completion_tokens")]
    gen = [r["generation_s"] for r in results if r.get("generation_s") is not None]
    retr = [r["retrieval_s"] for r in results if r.get("retrieval_s") is not None]
    tps = [r["completion_tokens"] / r["generation_s"]
           for r in results
           if r.get("completion_tokens") and r.get("generation_s")]

    return {
        "name": args.name,
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "wall_s": round(wall, 3),
        "rps": round(len(results) / wall, 3) if wall > 0 else None,
        "ok": len(latencies),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(results), 4) if results else None,
        "latency_s": describe(latencies),
        "latency_p90_s": round(percentile(latencies, 0.90), 3) if latencies else None,
        "latency_p99_s": round(percentile(latencies, 0.99), 3) if latencies else None,
        "retrieval_s": describe(retr),
        "generation_s": describe(gen),
        "tokens_per_s": describe(tps),
        "completion_tokens": describe(tokens),
        "errors_sample": [e.get("error") for e in errors[:10]],
        "per_request": [
            {"ok": r["ok"], "latency_s": round(r["latency_s"], 3), "error": r.get("error")}
            for r in results
        ],
    }


def plot(payload: dict, out_png: Path) -> None:
    if not HAS_MPL:
        print("[load] matplotlib недоступен — графики пропущены (uv pip install matplotlib)")
        return
    lat = [r["latency_s"] for r in payload.get("per_request", []) if r.get("ok")]
    if not lat:
        print("[load] нет успешных запросов — графики пропущены")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(range(1, len(lat) + 1), lat, marker=".", linewidth=1, color="#1f77b4")
    axes[0].set_title("Латентность по запросам")
    axes[0].set_xlabel("Номер запроса")
    axes[0].set_ylabel("Латентность, с")
    axes[0].grid(alpha=0.3)

    axes[1].hist(lat, bins=min(15, max(5, len(lat) // 2)), color="#2ca02c", alpha=0.8)
    axes[1].set_title("Распределение латентности")
    axes[1].set_xlabel("Латентность, с")
    axes[1].set_ylabel("Частота")
    axes[1].grid(alpha=0.3)

    d = payload.get("latency_s", {})
    labels = ["mean", "p50", "p90", "p95", "p99", "max"]
    values = [d.get("mean"), d.get("p50"), payload.get("latency_p90_s"),
              d.get("p95"), payload.get("latency_p99_s"), d.get("max")]
    values = [v if v is not None else 0 for v in values]
    axes[2].bar(labels, values, color=["#1f77b4", "#17becf", "#ff7f0e", "#d62728", "#8c564b", "#7f7f7f"])
    axes[2].set_title("Перцентили латентности")
    axes[2].set_ylabel("Латентность, с")
    axes[2].grid(alpha=0.3, axis="y")
    for i, v in enumerate(values):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    title = (
        f"Нагрузочный тест: {payload.get('name')} — "
        f"{payload.get('requests')} запросов, конкурентность {payload.get('concurrency')}, "
        f"{payload.get('rps')} RPS"
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[load] график: {out_png}")


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else RESULTS_DIR / f"load_{args.name}.json"
    out_png = out_json.with_suffix(".png")

    if args.plot_only:
        payload = json.loads(Path(args.plot_only).read_text(encoding="utf-8"))
        plot(payload, Path(args.plot_only).with_suffix(".png"))
        return 0

    questions = [row["question"] for row in load_qa(args.qa)]
    if not questions:
        raise SystemExit(f"Пустой golden-набор: {args.qa}")

    payload = run_load(args, questions)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[load] итоги:")
    print(f"  RPS={payload['rps']}, ошибок={payload['errors']}/{payload['requests']}")
    print(f"  latency  mean={payload['latency_s']['mean']} p95={payload['latency_s']['p95']} p99={payload['latency_p99_s']}")
    print(f"  tokens/s mean={payload['tokens_per_s']['mean']}")
    print(f"[load] сохранено: {out_json}")
    plot(payload, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
