"""RAG-ассистент: retrieval -> контекст с источниками -> LLM -> grounding.

Каждый запрос дополнительно трейсится в Langfuse (опционально).
Ответ проходит постобработку (grounding): каждая ссылка сверяется с реальными
метаданными извлечённых чанков, выдуманные источники удаляются.
"""

from __future__ import annotations

import dataclasses
import time

from llama_index.core.llms import ChatMessage, MessageRole

from normdoc.config import LLM_MODEL, LLM_PROVIDER, TOP_K
from normdoc.indexing import load_index, retrieve as index_retrieve
from normdoc.llm import build_llm
from normdoc.monitoring import get_tracer
from normdoc.prompts import (
    SYSTEM_PROMPT,
    format_context,
    ground_citations,
    user_message,
)


@dataclasses.dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]
    context: str
    dropped_citations: list[str]
    latency_s: float = 0.0
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    usage: dict | None = None
    trace: dict | None = None


class NormDocAssistant:
    """Ассистент по нормативным документам с системой цитирования."""

    def __init__(self, collection_name: str, top_k: int = TOP_K) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.index = load_index(collection_name=collection_name)
        self.llm = build_llm()
        self.tracer = get_tracer()

    def retrieve(self, query: str) -> list:
        return index_retrieve(
            query, top_k=self.top_k, collection_name=self.collection_name
        )

    @staticmethod
    def _source_dicts(nodes: list) -> list[dict]:
        sources: list[dict] = []
        for node_with_score in nodes:
            node = node_with_score.node
            sources.append(
                {
                    "metadata": node.metadata,
                    "score": getattr(node_with_score, "score", None),
                    "text": node.text,
                }
            )
        return sources

    def answer(self, query: str) -> RAGAnswer:
        started = time.monotonic()
        handle = self.tracer.trace("rag_answer", query)

        handle.current_span("retrieve")
        nodes = self.retrieve(query)
        retrieval_s = time.monotonic() - started
        handle.record("retrieve", input=query, metadata={"top_k": len(nodes)})

        context = format_context(nodes)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_message(context, query, nodes)),
        ]

        gen_started = time.monotonic()
        handle.current_span("generate")
        response = self.llm.chat(messages)
        generation_s = time.monotonic() - gen_started
        raw_answer = str(response.message.content).strip()
        usage = getattr(self.llm, "last_usage", None) or {}
        handle.record(
            "generate",
            input=user_message(context, query, nodes)[:8000],
            output=raw_answer,
            metadata={
                "chunks": len(nodes),
                "model": LLM_MODEL,
                "provider": LLM_PROVIDER,
            },
            usage=usage,
        )

        grounded, dropped = ground_citations(raw_answer, nodes)
        latency = time.monotonic() - started

        handle.finish(
            output=grounded,
            metadata={"chunks": len(nodes), "dropped": dropped},
            usage=usage,
        )

        return RAGAnswer(
            answer=grounded,
            sources=self._source_dicts(nodes),
            context=context,
            dropped_citations=dropped,
            latency_s=round(latency, 2),
            retrieval_s=round(retrieval_s, 3),
            generation_s=round(generation_s, 3),
            usage=usage,
            trace=handle.result(),
        )


def build_assistant(collection_name: str, top_k: int = TOP_K) -> NormDocAssistant:
    return NormDocAssistant(collection_name=collection_name, top_k=top_k)