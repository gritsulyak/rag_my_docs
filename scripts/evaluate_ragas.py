#!/usr/bin/env python
"""Оценка RAG-пайплайна метриками RAGAS.

Метрики:
- faithfulness       — опирается ли ответ на извлечённый контекст;
- context_precision  — релевантность извлечённого контекста (с эталоном);
- context_recall     — полнота контекста относительно эталонного ответа.

Судьёй выступает локальная LLM (bonsai или Meno-Lite) через llama.cpp,
эмбеддинги — локальные (e5-small / bge-m3). Выбор задаётся профилями
LLM_PROFILE / EMBEDDING_PROFILE (см. configs/).

Примеры:
    uv run python scripts/evaluate_ragas.py                       # профиль из .env
    LLM_PROFILE=meno EMBEDDING_PROFILE=e5_small \
        uv run python scripts/evaluate_ragas.py --out results/ragas_meno.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from normdoc import ragas_compat

ragas_compat.install_ragas_compat()

import requests  # noqa: E402
from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
)
from ragas.run_config import RunConfig  # noqa: E402

from normdoc.assistant import build_assistant  # noqa: E402
from normdoc.config import (  # noqa: E402
    COLLECTION_NAME,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    PROJECT_ROOT,
    QA_DIR,
    RESULTS_DIR,
    TOP_K,
)
from normdoc.indexing import make_embed_model  # noqa: E402
from normdoc.qa import load_qa  # noqa: E402

# Пороговые значения quality gate (плюс-требование «объективная оценка RAGAS»).
# Конфигурируются переменными окружения, значения по умолчанию — из опыта прогонов.
QUALITY_GATE = {
    "faithfulness": float(os.getenv("THRESH_FAITHFULNESS", "0.70")),
    "llm_context_precision_with_reference": float(os.getenv("THRESH_CONTEXT_PRECISION", "0.40")),
    "context_recall": float(os.getenv("THRESH_CONTEXT_RECALL", "0.60")),
}

_ROLE_MAP = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}


class ChatLlamaCpp(BaseChatModel):
    """Langchain ChatModel поверх llama.cpp (OpenAI-совместимый API) — судья RAGAS."""

    base_url: str = LLM_BASE_URL
    model: str = LLM_MODEL
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = 2048
    timeout: float = LLM_TIMEOUT

    @property
    def _llm_type(self) -> str:
        return "llamacpp"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": _ROLE_MAP.get(m.type, "user"), "content": m.content} for m in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        text = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", default=str(QA_DIR / "golden_qa.jsonl"))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число вопросов (0 = все)")
    parser.add_argument("--out", default=str(RESULTS_DIR / "ragas.json"))
    parser.add_argument("--judge-url", default=os.getenv("JUDGE_URL", "http://127.0.0.1:8081/v1"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "meno-lite"))
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Включить quality gate: ненулевой код возврата при недостижении порогов RAGAS",
    )
    return parser.parse_args()


def gate_result(metrics: dict) -> tuple[bool, dict]:
    """Сверяет агрегированные метрики с порогами; возвращает (passed, детали)."""
    details: dict = {}
    passed = True
    for key, threshold in QUALITY_GATE.items():
        value = metrics.get(key)
        ok = value is not None and float(value) >= threshold
        details[key] = {"value": value, "threshold": threshold, "ok": ok}
        passed = passed and ok
    return passed, details


def main() -> int:
    args = parse_args()
    rows = load_qa(args.qa)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[ragas] вопросов: {len(rows)}, коллекция: {args.collection}")

    assistant = build_assistant(collection_name=args.collection, top_k=args.top_k)

    samples: list[SingleTurnSample] = []
    perf: list[dict] = []
    for i, row in enumerate(rows, start=1):
        started = time.monotonic()
        result = assistant.answer(row["question"])
        latency = time.monotonic() - started
        contexts = [src["text"] for src in result.sources]
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=result.answer,
                retrieved_contexts=contexts,
                reference=row["ground_truth"],
            )
        )
        perf.append(
            {
                "id": row["id"],
                "latency_s": round(latency, 2),
                "retrieval_s": result.retrieval_s,
                "generation_s": result.generation_s,
                "chunks": len(contexts),
                "answer_chars": len(result.answer),
            }
        )
        print(f"[ragas] {i}/{len(rows)} {row['id']}: {latency:.1f} c, чанков={len(contexts)}")

    judge = LangchainLLMWrapper(
        ChatLlamaCpp(
            model=args.judge_model,
            base_url=args.judge_url,
            max_tokens=min(LLM_MAX_TOKENS, 1024),
        )
    )
    embeddings = LangchainEmbeddingsWrapper(make_embed_model())
    print(f"[ragas] судья: {args.judge_model} @ {args.judge_url}")

    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), ContextRecall()]

    print("[ragas] запускаю оценку (локальный судья, это может занять время)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=900, max_workers=2),
    )

    scores = {k: (round(float(v), 4) if v == v else None) for k, v in result._repr_dict.items()}
    aggregate = {k: round(float(v), 4) for k, v in result.items()} if hasattr(result, "items") else scores

    gate_passed, gate_details = gate_result(aggregate)
    mean = lambda key: round(sum(p[key] for p in perf) / len(perf), 3) if perf else None  # noqa: E731

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": args.collection,
        "llm_model": LLM_MODEL,
        "judge_model": args.judge_model,
        "n_questions": len(rows),
        "metrics": aggregate,
        "latency_s_mean": mean("latency_s"),
        "retrieval_s_mean": mean("retrieval_s"),
        "generation_s_mean": mean("generation_s"),
        "gate": {"enabled": args.gate, "passed": gate_passed if args.gate else None, "details": gate_details},
        "per_question": perf,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ragas] агрегированные метрики:")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print(f"[ragas] quality gate: {'ПРОЙДЕН' if gate_passed else 'ПРОВАЛЕН'} "
          + " ".join(f"{k}={v['value']}/{v['threshold']}" for k, v in gate_details.items()))
    print(f"[ragas] сохранено: {args.out}")
    if args.gate and not gate_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
