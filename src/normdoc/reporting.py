"""Сборка markdown-отчёта: RAGAS-метрики + perf-тесты + тесты + каталог моделей.

Модуль содержит только чистые функции форматирования, поэтому легко тестируется
и переиспользуется скриптом ``scripts/report.py``.
"""

from __future__ import annotations

from pathlib import Path

from normdoc.model_catalog import ModelInfo


def _fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def ragas_table(results: dict) -> str:
    """Таблица RAGAS по конфигурациям из results/comparison_ragas.json."""
    metric_keys = [
        ("faithfulness", "Faithfulness"),
        ("llm_context_precision_with_reference", "ContextPrecision"),
        ("context_recall", "ContextRecall"),
    ]
    lines = [
        "| Конфигурация | LLM | Эмбеддинги | Реранкер | Чанкинг | "
        + " | ".join(title for _, title in metric_keys)
        + " | Ср. латентность, с |",
        "|---|---|---|---|---|" + "---|" * (len(metric_keys) + 1),
    ]
    configs = results.get("configs", {})
    if not configs:
        return "_(нет данных: сначала запустите scripts/compare_configs.py)_"
    for name, cfg in configs.items():
        metrics = cfg.get("metrics", {})
        values = " | ".join(_fmt(metrics.get(key)) for key, _ in metric_keys)
        lines.append(
            f"| `{name}` | {cfg.get('llm', '—')} | {cfg.get('embeddings', '—')} "
            f"| {'да' if cfg.get('rerank') else 'нет'} | {cfg.get('chunking', '—')} "
            f"| {values} | {_fmt(cfg.get('latency_s_mean'), 2)} |"
        )
    return "\n".join(lines)


def ragas_single_section(run: dict | None) -> str:
    """Сводка контрольного одиночного прогона RAGAS на golden-наборе.

    Формат входа — ``results/ragas_golden.json`` (выход scripts/evaluate_ragas.py).
    """
    if not run:
        return ""
    metrics = run.get("metrics", {})
    gate = run.get("gate", {}) or {}
    details = gate.get("details") or {}
    lines = [
        f"**Контрольный прогон** на golden-наборе "
        f"(`data/qa/golden_qa.jsonl`, вопросов: {run.get('n_questions', '—')}): "
        f"Faithfulness **{_fmt(metrics.get('faithfulness'))}**, "
        f"ContextPrecision **{_fmt(metrics.get('llm_context_precision_with_reference'))}**, "
        f"ContextRecall **{_fmt(metrics.get('context_recall'))}**; "
        f"судья — `{run.get('judge_model', '—')}`, LLM — `{run.get('llm_model', '—')}`, "
        f"коллекция `{run.get('collection', '—')}`, "
        f"средняя латентность {_fmt(run.get('latency_s_mean'), 2)} с.",
    ]
    if details:
        if gate.get("passed") is True:
            verdict = "**ПРОЙДЕН**"
        elif gate.get("passed") is False:
            verdict = "**НЕ ПРОЙДЕН**"
        else:
            ok = all(bool(v.get("ok")) for v in details.values())
            verdict = "пороги соблюдены" if ok else "пороги не соблюдены"
        thresholds = "; ".join(
            f"{key} {_fmt(val.get('value'))}/{val.get('threshold')}"
            for key, val in details.items()
        )
        lines += ["", f"Quality gate: {verdict} — {thresholds}."]
    return "\n".join(lines)


def perf_table(perfs: list[dict]) -> str:
    """Таблица perf-тестов по конфигурациям из results/perf_*.json."""
    if not perfs:
        return "_(нет данных: сначала запустите scripts/benchmark_rag.py)_"
    lines = [
        "| Конфигурация | LLM | Коллекция | Реранкер | Ретрив p50, с | Ретрив p95, с "
        "| Генерация p50, с | Итого p50, с | ток/с | Ответ, симв. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for perf in perfs:
        retrieval = perf.get("retrieval_s", {})
        generation = perf.get("generation_s", {})
        total = perf.get("total_s", {})
        tps = perf.get("tokens_per_s", {}).get("mean")
        chars = perf.get("answer_chars", {}).get("mean")
        lines.append(
            f"| `{perf.get('name', '—')}` | {perf.get('llm_model', '—')} | {perf.get('collection', '—')} "
            f"| {'да' if perf.get('rerank') else 'нет'} | {_fmt(retrieval.get('p50'))} "
            f"| {_fmt(retrieval.get('p95'))} | {_fmt(generation.get('p50'))} | {_fmt(total.get('p50'))} "
            f"| {_fmt(tps, 2)} | {_fmt(chars, 0)} |"
        )
    return "\n".join(lines)


def catalog_table(models: list[ModelInfo]) -> str:
    """Каталог рекомендованных моделей в виде markdown-таблицы."""
    lines = [
        "| Модель | Организация | Тип | Параметры | Контекст | Лицензия | HuggingFace | Запуск |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        params = f"{m.params_b}B"
        if m.active_b:
            params += f" (активных {m.active_b}B)"
        hf = f"[{m.hf}](https://huggingface.co/{m.hf})" if m.hf else "—"
        lines.append(
            f"| {m.name} | {m.org} | {m.kind} | {params} | {m.context} | {m.license} | {hf} | {m.serve} |"
        )
    return "\n".join(lines)


def tests_section(pytest_text: str | None) -> str:
    """Короткая сводка по последнему прогону pytest."""
    if not pytest_text:
        return "_(нет data: сначала запустите scripts/run_tests.py)_"
    lines = [line.strip() for line in pytest_text.splitlines() if line.strip()]
    summary = lines[-1] if lines else "—"
    failed = [line for line in lines if line.startswith("FAILED") or " error" in line.lower()]
    out = [f"Итог последнего прогона: **{summary}**"]
    if failed:
        out.append("\nПроблемные тесты:\n")
        out.extend(f"- `{line}`" for line in failed[:20])
    return "\n".join(out)


def load_table(loads: list[dict]) -> str:
    """Таблица нагрузочных тестов по конфигурациям из results/load_*.json."""
    if not loads:
        return "_(нет данных: сначала запустите scripts/load_test.py)_"
    lines = [
        "| Конфигурация | Запросов | Конкурентность | RPS | p50, с | p95, с | p99, с | Ошибок | ток/с |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for load in loads:
        lat = load.get("latency_s", {})
        lines.append(
            f"| `{load.get('name', '—')}` | {load.get('requests', '—')} | {load.get('concurrency', '—')} "
            f"| {_fmt(load.get('rps'), 2)} | {_fmt(lat.get('p50'), 2)} | {_fmt(lat.get('p95'), 2)} "
            f"| {_fmt(load.get('latency_p99_s'), 2)} | {load.get('errors', '—')} "
            f"| {_fmt(load.get('tokens_per_s', {}).get('mean'), 2)} |"
        )
    images = [load.get("_png") for load in loads if load.get("_png")]
    if images:
        lines.append("")
        lines.append("Графики латентности (латентность по запросам / распределение / перцентили):")
        lines.append("")
        lines.extend(f"![Нагрузочный тест {load.get('name')}]({Path(load['_png']).name})" for load in loads if load.get("_png"))
    return "\n".join(lines)


def langfuse_section(langfuse: dict | None) -> str:
    """Секция мониторинга: последние трейсы Langfuse и latency спанов."""
    if not langfuse or not langfuse.get("available"):
        reason = (langfuse or {}).get("reason", "Langfuse не запущен")
        return (
            "_(трейсы Langfuse недоступны: "
            f"{reason}. Поднимите self-hosted Langfuse: `docker compose --profile monitoring up -d`, "
            "затем `uv run python scripts/langfuse_report.py`)_"
        )
    lines = [
        f"Хост: `{langfuse.get('host')}`, трейсов: **{langfuse.get('count')}**, "
        f"средняя латентность: **{_fmt(langfuse.get('latency_s_mean'), 2)} с**.",
        "",
        "| Trace | Вопрос | Латентность, с | Спаны (retrieve / generate, с) | Чанков |",
        "|---|---|---|---|---|",
    ]
    for tr in langfuse.get("traces", [])[:20]:
        spans = tr.get("spans", [])
        span_str = " / ".join(
            f"{s.get('name')}: {_fmt(s.get('latency_s'), 2)}" for s in spans if s.get("latency_s") is not None
        ) or "—"
        chunks = next((s.get("chunks") for s in spans if s.get("chunks")), None)
        question = (tr.get("input") or "").replace("\n", " ")[:80]
        lines.append(
            f"| `{tr.get('id', '')}` | {question} | {_fmt(tr.get('latency_s'), 2)} "
            f"| {span_str} | {chunks if chunks is not None else '—'} |"
        )
    return "\n".join(lines)


def docker_section(compose_info: dict | None = None) -> str:
    """Краткое описание docker-compose профилей железа и сервисов."""
    info = compose_info or {}
    lines = [
        "Приложение и инфраструктура упакованы в `docker-compose.yml`. Профили:",
        "",
        "| Профиль | Назначение | Бэкенд llama.cpp |",
        "|---|---|---|",
        "| `ai395` | AMD Ryzen AI MAX+ 395 / Radeon 8060S | Vulkan, сборка из исходников (`/dev/dri`,`/dev/kfd`) |",
        "| `amd7840hs` | AMD Ryzen 7 7840HS / Radeon 780M (iGPU) | Vulkan, предсобранный PrismML (`/dev/dri`) |",
        "| `rtx3060` | NVIDIA RTX 3060 (отдельный бокс), sm_86 | CUDA (nvidia runtime) |",
        "| `rocm` | AMD через ROCm/HIP (gfx1151) | HIP/ROCm |",
        "| `cpu` | Любая машина без GPU | CPU |",
        "| `monitoring` | Self-hosted Langfuse (Postgres, ClickHouse, Redis, MinIO) | — |",
        "| `tools` | Разовые задачи: индексация, RAGAS, load-тесты | — |",
        "",
        "Запуск (пример для AI MAX+ 395 с мониторингом):",
        "",
        "```bash",
        "docker compose --profile ai395 --profile monitoring up -d --build",
        "```",
    ]
    if info.get("services"):
        lines += ["", "Активные сервисы: " + ", ".join(f"`{s}`" for s in info["services"]) + "."]
    return "\n".join(lines)


def build_report(
    *,
    comparison: dict,
    perfs: list[dict],
    pytest_text: str | None,
    models: list[ModelInfo],
    environment: dict,
    corpus: dict,
    loads: list[dict] | None = None,
    langfuse: dict | None = None,
    compose_info: dict | None = None,
    ragas_single: dict | None = None,
) -> str:
    """Собирает полный markdown-отчёт."""
    env_lines = [f"- **{k}**: {v}" for k, v in environment.items() if v not in (None, "")]
    corpus_lines = [f"- **{k}**: {v}" for k, v in corpus.items() if v not in (None, "")]
    single_section = ragas_single_section(ragas_single)
    return "\n\n".join(
        [
            "# RAG по нормативным документам — отчёт о качестве и производительности",
            "\n".join(
                [
                    "## 1. Стенд",
                    *env_lines,
                    "",
                    "## 2. Корпус",
                    *corpus_lines,
                ]
            ),
            "\n".join(
                [
                    "## 3. Качество ответов (RAGAS)",
                    "Судья — локальная LLM, эмбеддинги — локальные. Faithfulness — опора ответа "
                    "на контекст, ContextPrecision — доля релевантных чанков, ContextRecall — полнота "
                    "контекста относительно эталона.",
                    "",
                    ragas_table(comparison),
                    *(["", single_section] if single_section else []),
                ]
            ),
            "\n".join(
                [
                    "## 4. Производительность (perf-тесты)",
                    "Замеры латентности ретривера и генерации, скорости генерации (ток/с) и длины ответа.",
                    "",
                    perf_table(perfs),
                ]
            ),
            "\n".join(
                [
                    "## 5. Нагрузочное тестирование инференса",
                    "HTTP-нагрузка на `/answer` под конкурентностью: RPS, перцентили латентности, "
                    "ошибки, токены/с. Графики строятся `scripts/load_test.py`.",
                    "",
                    load_table(loads or []),
                ]
            ),
            "\n".join(["## 6. Мониторинг Langfuse (трейсы и latency)", langfuse_section(langfuse)]),
            "\n".join(["## 7. Тесты", tests_section(pytest_text)]),
            "\n".join(
                [
                    "## 8. Каталог рекомендованных РФ-моделей для локального RAG",
                    catalog_table(models),
                ]
            ),
            "\n".join(["## 9. Инфраструктура и сервинг (Docker)", docker_section(compose_info)]),
            "\n".join(
                [
                    "## 10. Воспроизведение",
                    "```bash",
                    "# 1) поднять стек (выберите профиль железа)",
                    "docker compose --profile ai395 --profile monitoring up -d --build",
                    "",
                    "# 2) корпус и индекс",
                    "docker compose --profile tools run --rm app python scripts/prepare_docs.py",
                    "docker compose --profile tools run --rm app python scripts/build_index.py --reset \\",
                    "    --collection normdoc_e5_small",
                    "",
                    "# 3) качество (RAGAS) + производительность по конфигурациям",
                    "docker compose --profile tools run --rm app python scripts/compare_configs.py --limit 6",
                    "",
                    "# 4) нагрузочный тест с графиками",
                    "docker compose --profile tools run --rm app python scripts/load_test.py \\",
                    "    --url http://app:8000 --requests 40 --concurrency 4",
                    "",
                    "# 5) экспорт трейсов Langfuse и сборка отчёта",
                    "docker compose --profile tools run --rm app python scripts/langfuse_report.py",
                    "docker compose --profile tools run --rm app python scripts/report.py",
                    "```",
                ]
            ),
        ]
    )
