"""Реранкер bge-reranker-v2-m3 (cross-encoder) для переупорядочивания кандидатов.

Ретривер (bi-encoder) быстро достаёт широкий пул кандидатов (RETRIEVE_CANDIDATES),
после чего cross-encoder попарно оценивает «вопрос↔чанк» и оставляет TOP_K самых
релевантных. Это заметно повышает точность контекста, но требует загрузки ещё
одной модели (~2.3 ГБ) и добавляет латентность.

Модуль лениво загружает модель: если RERANK_ENABLED=0, он не импортирует
sentence-transformers и работает как no-op.
"""

from __future__ import annotations

import time

from normdoc.config import (
    RERANK_BATCH,
    RERANK_DEVICE,
    RERANK_ENABLED,
    RERANK_MODEL,
)


def _pick_device() -> str:
    if RERANK_DEVICE != "auto":
        return RERANK_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Reranker:
    """Cross-encoder реранкер с ленивой загрузкой."""

    def __init__(self, model_name: str = RERANK_MODEL, batch_size: int = RERANK_BATCH) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = _pick_device()
            print(f"[reranker] загружаю {self.model_name} на {device}")
            self._model = CrossEncoder(self.model_name, device=device, max_length=512)
        return self._model

    def rerank(self, query: str, nodes: list, top_n: int) -> list:
        """Возвращает top_n кандидатов, отсортированных cross-encoder'ом.

        score в NodeWithScore заменяется на скор реранкера (для логов/трасс).
        """
        if not nodes:
            return []
        model = self._load()
        pairs = [(query, nw.node.get_content()) for nw in nodes]
        started = time.monotonic()
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        elapsed = time.monotonic() - started
        # `predict` может вернуть 1-D или 2-D (для многоклассовых) массив.
        scored = []
        for nw, score in zip(nodes, scores):
            value = float(score[-1]) if hasattr(score, "__len__") and not isinstance(score, (int, float)) else float(score)
            try:
                nw.score = value
            except Exception:
                pass
            scored.append(nw)
        scored.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
        print(f"[reranker] {len(nodes)} кандидатов -> {top_n} за {elapsed:.2f} c")
        return scored[:top_n]


_reranker: Reranker | None = None


def get_reranker() -> Reranker | None:
    """Возвращает singleton-реранкер или None, если он выключен."""
    global _reranker
    if not RERANK_ENABLED:
        return None
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
