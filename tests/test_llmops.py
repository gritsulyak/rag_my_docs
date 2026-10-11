"""Тесты LLMOps-слоя: golden-QA, отчётность, мониторинг, HTTP-API, провайдеры LLM.

Не требуют реальных LLM/Qdrant: API-тест подменяет ассистента фейком,
мониторинг проверяется в no-op и offline-режиме, провайдеры — через
monkeypatch ``requests.post``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from normdoc import llm as llm_mod
from normdoc import monitoring
from normdoc import reporting
from normdoc.assistant import RAGAnswer
from normdoc.model_catalog import CATALOG
from normdoc.qa import GOLDEN_QA_PATH, describe, load_qa, percentile


# --------------------------------------------------------------------------- qa
class TestGoldenQa:
    def test_load_qa_real_file(self):
        rows = load_qa(GOLDEN_QA_PATH)
        assert len(rows) >= 10
        for row in rows:
            assert set(row) >= {"id", "question", "ground_truth"}
            assert row["question"].strip()
            assert row["ground_truth"].strip()

    def test_ids_unique(self):
        ids = [r["id"] for r in load_qa(GOLDEN_QA_PATH)]
        assert len(ids) == len(set(ids))

    def test_percentile_interpolates(self):
        assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
        assert percentile([], 0.5) is None
        assert percentile([42], 0.9) == 42

    def test_describe(self):
        stats = describe([1.0, 2.0, 3.0])
        assert stats["n"] == 3
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["min"] == 1.0 and stats["max"] == 3.0
        assert describe([])["n"] == 0


# --------------------------------------------------------------------- reporting
class TestReporting:
    def test_ragas_table_has_all_metrics(self):
        comparison = {
            "configs": {
                "baseline": {
                    "llm": "bonsai",
                    "embeddings": "e5_small",
                    "rerank": False,
                    "chunking": "semantic",
                    "latency_s_mean": 12.0,
                    "metrics": {
                        "faithfulness": 0.8,
                        "llm_context_precision_with_reference": 0.5,
                        "context_recall": 0.7,
                    },
                }
            }
        }
        md = reporting.ragas_table(comparison)
        assert "baseline" in md and "0.800" in md and "Faithfulness" in md

    def test_ragas_table_empty(self):
        assert "нет данных" in reporting.ragas_table({})

    def test_perf_table_empty(self):
        assert "нет данных" in reporting.perf_table([])

    def test_ragas_single_section(self):
        run = {
            "n_questions": 3,
            "judge_model": "meno-lite",
            "llm_model": "bonsai",
            "collection": "normdoc_e5_small",
            "latency_s_mean": 23.0,
            "metrics": {
                "faithfulness": 1.0,
                "llm_context_precision_with_reference": 0.956,
                "context_recall": 1.0,
            },
            "gate": {
                "passed": True,
                "details": {
                    "faithfulness": {"value": 1.0, "threshold": 0.7, "ok": True},
                    "context_recall": {"value": 1.0, "threshold": 0.6, "ok": True},
                },
            },
        }
        md = reporting.ragas_single_section(run)
        assert "Контрольный прогон" in md and "1.000" in md and "ПРОЙДЕН" in md
        assert reporting.ragas_single_section({}) == ""

    def test_load_table_with_graph(self):
        loads = [
            {
                "name": "docker",
                "requests": 40,
                "concurrency": 4,
                "rps": 0.7,
                "latency_s": {"p50": 12.0, "p95": 20.0},
                "latency_p99_s": 25.0,
                "errors": 0,
                "tokens_per_s": {"mean": 30.0},
                "_png": "/tmp/opencode/load_docker.png",
            }
        ]
        md = reporting.load_table(loads)
        assert "docker" in md and "RPS" in md
        assert "load_docker.png" in md

    def test_langfuse_unavailable(self):
        md = reporting.langfuse_section({"available": False, "reason": "нет ключей"})
        assert "недоступны" in md and "нет ключей" in md

    def test_langfuse_available(self):
        payload = {
            "available": True,
            "host": "http://localhost:3000",
            "count": 1,
            "latency_s_mean": 15.5,
            "traces": [
                {
                    "id": "abcdef012345",
                    "input": "Сколько классов?",
                    "latency_s": 15.5,
                    "spans": [
                        {"name": "retrieve", "latency_s": 0.4, "chunks": 6},
                        {"name": "generate", "latency_s": 15.1},
                    ],
                }
            ],
        }
        md = reporting.langfuse_section(payload)
        assert "abcdef01" in md and "retrieve: 0.40" in md and "6" in md

    def test_docker_section_lists_profiles(self):
        md = reporting.docker_section({"services": ["qdrant", "app"]})
        for token in ("ai395", "rtx3060", "rocm", "cpu", "monitoring"):
            assert token in md

    def test_build_report_smoke(self):
        md = reporting.build_report(
            comparison={},
            perfs=[],
            pytest_text="19 passed in 57.69s",
            models=CATALOG,
            environment={"ОС": "Linux", "RAM, ГБ": 27},
            corpus={"Документов": 13},
            loads=[],
            langfuse={"available": False, "reason": "нет"},
            compose_info={},
        )
        assert "# RAG по нормативным документам" in md
        assert "RAGAS" in md and "Langfuse" in md and "Docker" in md


# -------------------------------------------------------------------- monitoring
class TestMonitoringNoop:
    def test_handle_without_langfuse_is_safe(self):
        handle = monitoring.TraceHandle()
        handle.current_span("retrieve")
        handle.record("retrieve", input="q", metadata={"top_k": 5})
        handle.current_span("generate")
        handle.record("generate", output="a", usage={"completion_tokens": 3}, metadata={"model": "x"})
        handle.finish("a", metadata={"latency_s": 1.0}, usage={"completion_tokens": 3})
        assert "latency_s" in handle.result()

    def test_tracer_disabled_without_keys(self):
        tracer = monitoring.LangfuseTracer(enabled=False, public_key="", secret_key="", host="")
        handle = tracer.trace("rag_answer", "вопрос")
        assert handle is not None
        handle.finish("ответ")

    def test_usage_mapping(self):
        assert monitoring._usage_details(None) is None
        mapped = monitoring._usage_details({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        assert mapped == {"input": 10, "output": 5, "total": 15}

    def test_local_jsonl_sink(self, tmp_path, monkeypatch):
        path = tmp_path / "traces.jsonl"
        monkeypatch.setattr(monitoring, "_TRACE_JSONL", str(path))
        handle = monitoring.TraceHandle()
        handle.current_span("retrieve")
        handle.record("retrieve", input="q", metadata={"chunks": 5})
        handle.current_span("generate")
        handle.record("generate", output="a", usage={"completion_tokens": 3}, metadata={"model": "x"})
        handle.finish("a", metadata={"chunks": 5}, usage={"completion_tokens": 3})

        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["name"] == "rag_answer"
        assert rec["latency_s"] >= 0
        assert [s["name"] for s in rec["spans"]] == ["retrieve", "generate"]


# -------------------------------------------------------------------------- api
class _FakeAssistant:
    collection_name = "normdoc_e5_small"

    def answer(self, query: str) -> RAGAnswer:
        return RAGAnswer(
            answer="Тестовый ответ [1].",
            sources=[
                {
                    "metadata": {"source": "gost.pdf", "page": 3},
                    "score": 0.91,
                    "text": "релевантный фрагмент",
                }
            ],
            context="ctx",
            dropped_citations=[],
            latency_s=1.23,
            retrieval_s=0.10,
            generation_s=1.13,
            usage={"completion_tokens": 7, "total_tokens": 20},
        )


class TestApi:
    @pytest.fixture()
    def client(self, monkeypatch):
        import normdoc.api as api

        monkeypatch.setattr(api, "get_assistant", lambda collection, top_k: _FakeAssistant())
        return TestClient(api.app)

    def test_health(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "collection" in data

    def test_answer(self, client):
        resp = client.post("/answer", json={"question": "Что такое архитектура?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"].startswith("Тестовый ответ")
        assert body["sources"][0]["page"] == 3
        assert body["latency_s"] == 1.23

    def test_answer_validation(self, client):
        assert client.post("/answer", json={"question": ""}).status_code == 422

    def test_metrics_exposed(self, client):
        client.post("/answer", json={"question": "вопрос"})
        text = client.get("/metrics").text
        assert "rag_requests_total" in text
        assert "rag_total_latency_seconds" in text

    def test_ui_page_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "/answer" in resp.text and "/ask" in resp.text and "<textarea" in resp.text

    def test_ui_alias_and_info(self, client):
        assert client.get("/ui").status_code == 200
        info = client.get("/info").json()
        assert "/ask" in info["endpoints"] and info["service"] == "normdoc-assistant"

    def test_ask_html_fallback(self, client):
        resp = client.get("/ask", params={"q": "Что такое архитектура?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "Тестовый ответ" in resp.text
        assert "gost.pdf" in resp.text and "Источники" in resp.text
        assert client.get("/ask").status_code == 422  # без q


# ----------------------------------------------------------------- llm providers
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestLlmDispatch:
    def test_openai_payload_has_reasoning_effort(self):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kw):
            captured["url"] = url
            captured["json"] = json
            return _FakeResp({"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}})

        orig = llm_mod.requests.post
        llm_mod.requests.post = fake_post
        try:
            text, usage = llm_mod.call_llamacpp(
                [{"role": "user", "content": "hi"}],
                base_url="http://x/v1",
                model="m",
                reasoning="medium",
            )
        finally:
            llm_mod.requests.post = orig
        assert text == "ok"
        assert captured["json"]["reasoning_effort"] == "medium"
        assert captured["url"].endswith("/chat/completions")

    def test_call_llm_routes_gigachat(self, monkeypatch):
        called = {}

        def fake_giga(messages, **kwargs):
            called["messages"] = messages
            called["kwargs"] = kwargs
            return "giga", {"completion_tokens": 2}

        monkeypatch.setattr(llm_mod, "call_gigachat", fake_giga)
        text, usage = llm_mod.call_llm(
            [{"role": "user", "content": "привет"}],
            provider="gigachat",
            model="GigaChat",
            base_url="http://should-be-ignored",
        )
        assert text == "giga"
        # base_url не должен попасть в GigaChat-ветку.
        assert "base_url" not in called["kwargs"]
        assert called["kwargs"]["model"] == "GigaChat"

    def test_gigachat_token_cache(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(url, headers=None, data=None, timeout=None, verify=None, **kw):
            calls["n"] += 1
            return _FakeResp({"access_token": "tok", "expires_at": None})

        monkeypatch.setattr(llm_mod.requests, "post", fake_post)
        llm_mod._TOKEN_CACHE["value"] = None
        llm_mod._TOKEN_CACHE["expires_at"] = 0.0
        t1 = llm_mod._gigachat_access_token(credentials="abc", auth_url="http://a", scope="s", verify=False)
        t2 = llm_mod._gigachat_access_token(credentials="abc", auth_url="http://a", scope="s", verify=False)
        assert t1 == t2 == "tok"
        assert calls["n"] == 1, "второй вызов должен взять токен из кэша"
