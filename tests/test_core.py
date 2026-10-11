"""Тесты парсинга, чанкинга и системы цитирования.

Не требуют эмбеддингов/Qdrant/LLM — работают на подготовленных PDF из
data/raw/pdf (см. scripts/prepare_docs.py). Запуск: uv run python scripts/run_tests.py

Приватный корпус не публикуется в git, поэтому тесты, которым нужны реальные
документы (``data/raw/manifest.json`` + PDF), при его отсутствии ПРОПУСКАЮТСЯ
(``@requires_corpus``), а не падают — так CI проходит на чистом клоне.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from normdoc.config import MANIFEST_PATH, PDF_DIR
from normdoc.parsing import (
    load_all_chunks,
    load_manifest,
    load_ocr,
    parse_pdf,
    sanitize_links,
    split_text,
)
from normdoc.prompts import (
    allowed_citations,
    format_context,
    ground_citations,
    source_label,
)

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_JSON = ROOT / "data" / "chunks" / "chunks.json"

# Приватный корпус (PDF + manifest.json) не публикуется в git. Тесты, которым
# нужны реальные документы, на чистом клоне (CI) пропускаются, а не падают.
CORPUS_AVAILABLE = MANIFEST_PATH.exists()
requires_corpus = pytest.mark.skipif(
    not CORPUS_AVAILABLE,
    reason="приватный корпус не подготовлен (нет data/raw/manifest.json)",
)


def _manifest_by_label(label: str) -> dict:
    for entry in load_manifest()["files"]:
        if entry["label"] == label:
            return entry
    raise KeyError(label)


def _nodes(metas: list[dict]) -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text=f"текст{i}", metadata=meta))
        for i, meta in enumerate(metas)
    ]


class TestSanitizeLinks:
    def test_removes_urls_and_emails(self):
        text = "см. http://www.opengroup.org/togaf/ и www.gost.ru, e-mail a@b.ru"
        out = sanitize_links(text)
        assert "http" not in out
        assert "www." not in out
        assert "@" not in out
        assert "см." in out

    def test_keeps_plain_text(self):
        assert sanitize_links("ГОСТ Р 57100—2016") == "ГОСТ Р 57100—2016"


class TestChunking:
    def test_respects_size(self):
        text = " ".join(f"Предложение номер {i}." for i in range(200))
        parts = split_text(text, chunk_size=120, chunk_overlap=12)
        assert parts
        assert all(len(p) <= 200 for p in parts)  # допуск на склейку строк
        assert len(parts) > 1

    def test_overlap_present(self):
        text = "\n".join(f"Строка {i} с некоторым текстом для перекрытия." for i in range(50))
        parts = split_text(text, chunk_size=120, chunk_overlap=30)
        joined_ok = any(parts[i][-10:] in parts[i + 1] or parts[i + 1][:10] in parts[i] for i in range(len(parts) - 1))
        assert joined_ok

    @requires_corpus
    def test_load_all_chunks_metadata(self):
        chunks = load_all_chunks(chunk_size=400, chunk_overlap=50)
        assert chunks
        for chunk in chunks:
            meta = chunk.metadata
            assert meta["source"]
            assert meta["kind"]
            assert meta["page"] is None or meta["page"] >= 1
            assert meta["doc_type"] == "pdf"
            assert chunk.text.strip()


class TestPdfParsing:
    @requires_corpus
    def test_first_page_number(self):
        entry = _manifest_by_label("ГОСТ Р 57100—2016")
        pages = parse_pdf(
            PDF_DIR / entry["path"],
            label=entry["label"],
            title=entry.get("title", ""),
            kind="standard",
        )
        # Первая страница документа всегда печатная страница 1.
        assert pages[0].page == 1
        assert pages[0].physical_page == 1

    @requires_corpus
    def test_page_numbers_within_document(self):
        for entry in load_manifest()["files"]:
            pages = parse_pdf(
                PDF_DIR / entry["path"],
                label=entry["label"],
                title=entry.get("title", ""),
                kind=entry["kind"],
            )
            for page in pages:
                assert page.page is not None
                assert 1 <= page.page <= entry["pages"]

    def test_load_ocr_reads_sidecar(self, tmp_path, monkeypatch):
        import normdoc.parsing as parsing

        monkeypatch.setattr(parsing, "OCR_DIR", tmp_path)
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        (tmp_path / "scan.json").write_text(
            json.dumps({"pages": [{"physical": 1, "text": "РАСПОЗНАННЫЙ ТЕКСТ"}]},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_ocr(pdf) == {1: "РАСПОЗНАННЫЙ ТЕКСТ"}

    def test_load_ocr_missing_sidecar(self, tmp_path, monkeypatch):
        import normdoc.parsing as parsing

        monkeypatch.setattr(parsing, "OCR_DIR", tmp_path)
        assert load_ocr(tmp_path / "absent.pdf") == {}

    @requires_corpus
    def test_tables_serialized(self):
        entry = _manifest_by_label("ГОСТ Р 57100—2025")
        pages = parse_pdf(
            PDF_DIR / entry["path"], label=entry["label"], title="", kind="standard"
        )
        assert any("[Таблица]" in p.text for p in pages)

    @requires_corpus
    def test_figure_captions_detected(self):
        entry = _manifest_by_label("ГОСТ Р 57100—2025")
        pages = parse_pdf(
            PDF_DIR / entry["path"], label=entry["label"], title="", kind="standard"
        )
        assert any(p.figure for p in pages)


class TestCitations:
    def test_source_label(self):
        assert source_label({"source": "a.pdf", "page": 4}) == "a.pdf, стр. 4"
        assert source_label({"source": "a.pdf"}) == "a.pdf"

    def test_allowed_citations_unique(self):
        nodes = _nodes(
            [
                {"source": "a.pdf", "page": 1},
                {"source": "a.pdf", "page": 1},
                {"source": "b.pdf", "page": 2},
            ]
        )
        assert allowed_citations(nodes) == ["a.pdf, стр. 1", "b.pdf, стр. 2"]

    def test_format_context_includes_figure(self):
        nodes = _nodes([{"source": "a.pdf", "page": 3, "figure": "Рисунок 1 – Схема"}])
        context = format_context(nodes)
        assert "Рисунок 1" in context
        assert "Контекст 1" in context

    def test_grounding_keeps_exact(self):
        nodes = _nodes([{"source": "a.pdf", "page": 4}])
        out, dropped = ground_citations("Факт [a.pdf, стр. 4].", nodes)
        assert out == "Факт [a.pdf, стр. 4]."
        assert dropped == []

    def test_grounding_resolves_context_number(self):
        nodes = _nodes([{"source": "a.pdf", "page": 4}, {"source": "b.pdf", "page": 7}])
        out, _ = ground_citations("Факт [Контекст 2].", nodes)
        assert out == "Факт [b.pdf, стр. 7]."

    def test_grounding_cleans_nested_description(self):
        nodes = _nodes([{"source": "a.pdf", "page": 4}])
        out, dropped = ground_citations("Факт [Контекст 1 (Источник: a.pdf, стр. 4)].", nodes)
        assert out == "Факт [a.pdf, стр. 4]."
        assert dropped == []

    def test_grounding_drops_hallucinated(self):
        nodes = _nodes([{"source": "a.pdf", "page": 4}])
        out, dropped = ground_citations("Факт [выдумка.pdf, стр. 99].", nodes)
        assert "выдумка" not in out
        assert dropped == ["[выдумка.pdf, стр. 99]"]


class TestLlmPayload:
    def test_reasoning_effort_passed(self, monkeypatch):
        calls = {}

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        def fake_post(url, json=None, timeout=None):
            calls["url"] = url
            calls["json"] = json
            return FakeResp()

        monkeypatch.setattr("normdoc.llm.requests.post", fake_post)
        from normdoc.llm import call_llamacpp

        text, usage = call_llamacpp(
            [{"role": "user", "content": "привет"}], reasoning="none", max_tokens=64
        )
        assert text == "ok"
        assert calls["json"]["reasoning_effort"] == "none"
        assert calls["json"]["max_tokens"] == 64
        assert calls["url"].endswith("/chat/completions")


class TestChunksArtifact:
    def test_chunks_json_has_metadata(self):
        if not CHUNKS_JSON.exists():
            pytest.skip("нет data/chunks/chunks.json (сначала build_index.py)")
        payload = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        assert payload
        assert all(item["metadata"].get("source") for item in payload)
