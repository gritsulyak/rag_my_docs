"""Парсинг нормативных документов.

Особенности набора данных (национальные стандарты, приказы, регламенты):

- PDF: PyMuPDF. Страница -> текст + метаданные:
  * печатный номер страницы (из нижнего колонтитула «стр. N»);
  * заголовок раздела из верхнего колонтитула;
  * **таблицы** распознаются `page.find_tables()` и сериализуются отдельным
    блоком `[Таблица]` с разделителями `|`, чтобы не терять структуру колонок;
  * **подписи рисунков/диаграмм** («Рисунок N – …») детектируются рядом с
    изображениями и выносятся в metadata как `figure`, чтобы страницы с
    диаграммами находились по подписи (текст внутри векторных диаграмм PyMuPDF
    извлекает автоматически; для растровых диаграмм подпись — единственный
    текстовый сигнал);
  * все ссылки `http(s)://`, `www.`, email-адреса **вырезаются**
    (SANITIZE_LINKS=1), чтобы мусорные токены не попадали в Qdrant.
- Сканы без текстового слоя: текст берётся из sidecar OCR-файла
  `data/raw/ocr/<stub>.json`, подготовленного scripts/prepare_docs.py.
- RTF/ODT конвертируются в PDF на этапе prepare_docs.py, поэтому здесь всё
  единообразно обрабатывается как PDF.

Метаданные хранятся на двух уровнях — страницы (parse_pdf) и чанка
(parse_pdf_to_chunks), поэтому ссылка `[документ, стр. N]` всегда привязана
к фактической странице.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from normdoc.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE,
    CHUNK_TARGET_TOKENS,
    CHUNKING_MODE,
    CHUNKS_JSON,
    MANIFEST_PATH,
    OCR_DIR,
    PDF_DIR,
    PARSE_FIGURES,
    PARSE_TABLES,
    SANITIZE_LINKS,
    TOKENIZER_NAME,
)

# Полосы колонтитулов (доля высоты страницы).
HEADER_BAND = 0.11
FOOTER_BAND = 0.92

PRINTED_PAGE_RE = re.compile(r"стр\.\s*(\d+)", re.IGNORECASE)

# Подписи рисунков, схем, диаграмм.
FIGURE_RE = re.compile(
    r"^(Рисунок|Схема|Диаграмма|Фиг\.?)\s*([0-9]+|[А-ЯA-Z]-?[0-9])\s*([—–-]|\.|\))?\s*(.+)$",
    re.IGNORECASE,
)

# Мусорные ссылки: http(s)://, www., email.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _strip_url_artefacts(text: str) -> str:
    """После удаления ссылки убирает хвостовые скобки/кавычки, которые остались от линка."""
    return re.sub(r"\s+([)}\]„»\"])", r"\1", text)


def sanitize_links(text: str) -> str:
    """Удаляет http(s)/www/email из текста, оставляя окружающие слова как есть."""
    if not text:
        return text
    out = _EMAIL_RE.sub(" ", text)
    out = _URL_RE.sub(" ", out)
    out = _strip_url_artefacts(out)
    return re.sub(r" {2,}", " ", out).strip()


def load_manifest() -> dict:
    """Загружает data/raw/manifest.json (список файлов для индексации)."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Манифест {MANIFEST_PATH} не найден. Запустите scripts/prepare_docs.py"
        )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------


@dataclass
class PageContent:
    """Содержимое одной страницы PDF + извлечённые метаданные."""

    text: str
    source: str          # короткая метка документа (для цитат)
    title: str           # полное название документа
    kind: str            # standard | law | regulation | form | release_notes
    page: int | None     # печатный номер страницы (для цитирования)
    physical_page: int   # физический номер страницы в PDF
    header: str          # заголовок раздела из верхнего колонтитула
    figure: str | None = field(default=None, repr=False)

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "kind": self.kind,
            "page": self.page,
            "physical_page": self.physical_page,
            "header": self.header,
            "figure": self.figure,
            "doc_type": "pdf",
        }


@dataclass
class Chunk:
    """Фрагмент текста с жёстко привязанным словарём метаданных."""

    text: str
    metadata: dict


# ---------------------------------------------------------------------------
# Колонтитулы и тело страницы
# ---------------------------------------------------------------------------


def _join_lines(words: list[tuple]) -> str:
    """Схлопывает слова в строки по y-координате, затем в текст."""
    lines: list[list[tuple]] = []
    for word in sorted(words, key=lambda w: (round(w[1], 1), w[0])):
        if lines and abs(word[1] - lines[-1][-1][1]) < 2:
            lines[-1].append(word)
        else:
            lines.append([word])
    return "\n".join(
        " ".join(w[4] for w in sorted(line, key=lambda w: w[0])) for line in lines
    ).strip()


def _header_text(page: pymupdf.Page) -> str:
    """Текст в верхней полосе страницы (заголовок раздела / стандарта)."""
    words = page.get_text("words")
    band = page.rect.height * HEADER_BAND
    top_words = [w for w in words if w[3] <= band]
    return _join_lines(top_words)


def _printed_page(page: pymupdf.Page, page_count: int | None = None) -> int | None:
    """Печатный номер страницы из нижнего колонтитула.

    Номер должен быть валидным (1..page_count): иначе в колонтитуле можно
    принять за номер страницы год («… 2025») или номер редакции.
    """
    words = page.get_text("words")
    band = page.rect.height * FOOTER_BAND
    footer_words = [w for w in words if w[1] >= band]
    footer_text = " ".join(w[4] for w in footer_words)
    match = PRINTED_PAGE_RE.search(footer_text)
    if match:
        value = int(match.group(1))
        if page_count is None or 1 <= value <= page_count:
            return value
    numbers = [int(w[4]) for w in footer_words if w[4].isdigit()]
    if page_count is not None:
        numbers = [n for n in numbers if 1 <= n <= page_count]
    return numbers[-1] if numbers else None


def _table_serialization(page: pymupdf.Page) -> list[tuple[str, list[pymupdf.Rect]]]:
    """Извлекает таблицы страницы.

    Возвращает список (сериализованный_текст, rect_таблицы). Первая строка
    каждой таблицы считается шапкой. Сериализация: колонки через ' | ',
    строки через новые строки.
    """
    if not PARSE_TABLES:
        return []
    finder = page.find_tables()
    out: list[tuple[str, list[pymupdf.Rect]]] = []
    for tab in getattr(finder, "tables", []):
        rows = tab.extract()
        if not rows:
            continue
        rects: list[pymupdf.Rect] = []
        try:
            cells = getattr(tab, "cells", None)
            if cells:
                rects = [pymupdf.Rect(c) for c in cells]
        except Exception:
            rects = []
        if not rects:
            rects = [pymupdf.Rect(tab.bbox)]
        lines: list[str] = []
        for row in rows:
            cells_text = [str(c).strip() if c is not None else "" for c in row]
            lines.append(" | ".join(cells_text).strip(" |"))
        if any(lines):
            out.append(("[Таблица]\n" + "\n".join(lines), rects))
    return out


def _line_map(words: list[tuple]) -> list[tuple[float, str]]:
    """Строит строки (y0, текст) из слов: слова с разницей y < 2 склеиваются."""
    lines: list[tuple[float, str]] = []
    for word in sorted(words, key=lambda w: (round(w[1], 1), w[0])):
        if lines and abs(word[1] - lines[-1][0]) < 2:
            text = lines[-1][1] + " " + word[4]
            lines[-1] = (lines[-1][0], text)
        else:
            lines.append((word[1], word[4]))
    return lines


def _figure_captions(page: pymupdf.Page, words: list[tuple]) -> list[tuple[pymupdf.Rect, str]]:
    """Детектирует блоки изображений и подписи к ним («Рисунок 1 – …»).

    Возвращает пары (bbox_изображения, полный_текст_подписи). Диаграммы без
    векторов и подписей не индексируются как текст (для этого нужна VLM).
    """
    if not PARSE_FIGURES:
        return []

    images: list[pymupdf.Rect] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") in (1, 3):
            try:
                images.append(pymupdf.Rect(block["bbox"]))
            except Exception:
                continue

    caption_lines = [
        (y, text)
        for y, text in _line_map(words)
        if FIGURE_RE.match(text.strip())
    ]

    captions: list[tuple[pymupdf.Rect, str]] = []
    for img in images:
        below = [(y, m.group(1) + " " + m.group(4).strip()) for y, text in caption_lines for m in [FIGURE_RE.match(text.strip())] if m and y >= img.y1 - 2]
        if below:
            y, caption = min(below, key=lambda p: p[0])
            captions.append((img, caption))
    return captions


def _body_words_with_tables(
    page: pymupdf.Page,
) -> tuple[list[tuple], list[str]]:
    """Слова тела страницы (без колонтитулов и без текста таблиц) + сериализация таблиц."""
    words = page.get_text("words")
    header_band = page.rect.height * HEADER_BAND
    footer_band = page.rect.height * FOOTER_BAND
    body_words = [w for w in words if header_band < w[3] and w[1] < footer_band]

    serialized: list[str] = []
    table_rects: list[pymupdf.Rect] = []
    tables = _table_serialization(page)
    if tables:
        serialized = [t for t, _ in tables]
        for _, rects in tables:
            table_rects.extend(rects)

    if table_rects:
        kept: list[tuple] = []
        for w in body_words:
            rect = pymupdf.Rect(w[0], w[1], w[2], w[3])
            inside = any(rect.intersects(t) for t in table_rects)
            if not inside:
                kept.append(w)
        body_words = kept

    return body_words, serialized


def _sanitize(text: str) -> str:
    return sanitize_links(text) if SANITIZE_LINKS else text


# ---------------------------------------------------------------------------
# Основной парсер PDF
# ---------------------------------------------------------------------------


def parse_pdf(
    path: Path,
    *,
    label: str,
    title: str,
    kind: str,
    ocr_pages: dict[int, str] | None = None,
) -> list[PageContent]:
    """PDF -> список страниц с метаданными.

    Страницы с пустым телом пропускаются (титульники-картинки, болванки).
    Если для страницы нет извлечённого текста, но есть OCR sidecar - используем его.
    """
    ocr_pages = ocr_pages or {}
    pages: list[PageContent] = []

    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        for index, page in enumerate(doc, start=1):
            words = page.get_text("words")
            body_words, tables_text = _body_words_with_tables(page)
            body = _join_lines(body_words)
            if not body.strip():
                body = _sanitize(ocr_pages.get(index, "")).strip()
            if not body.strip():
                continue

            header = _header_text(page)
            header = _sanitize(header)

            captions = _figure_captions(page, words)

            figure_text = " | ".join(c for _, c in captions) or None
            full_text = _sanitize(body)
            if tables_text:
                full_text = full_text + "\n\n" + "\n\n".join(tables_text)

            printed = _printed_page(page, page_count)
            page_number = printed if printed is not None else index

            pages.append(
                PageContent(
                    text=full_text,
                    source=label,
                    title=title,
                    kind=kind,
                    page=page_number,
                    physical_page=index,
                    header=header,
                    figure=figure_text,
                )
            )
    return pages


def load_ocr(pdf_path: Path) -> dict[int, str]:
    """Подгружает sidecar OCR-текст data/raw/ocr/<stub>.json, если есть."""
    sidecar = OCR_DIR / f"{pdf_path.stem}.json"
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    pages = data.get("pages", []) if isinstance(data, dict) else data.get("pages", [])
    return {int(p["physical"]): p["text"] for p in pages if p.get("text")}


# ---------------------------------------------------------------------------
# Чанкинг
# ---------------------------------------------------------------------------


def _split_long_line(line: str, chunk_size: int) -> list[str]:
    """Длинную строку режем по границам предложений, склеивая короткие фразы."""
    sentences = re.split(r"(?<=[.!?…])\s+", line.strip())
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if buffer and len(candidate) > chunk_size:
            out.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        out.append(buffer)
    return out


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Разбиение на чанки по строкам (сохраняет границы таблиц) и предложениям.

    Каждая непустая строка становится «атомом»: строки таблиц не разрываются,
    длинные абзацы режутся по границам предложений. Чанки склеиваются с overlap.
    """
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= chunk_size:
            units.append(line)
        else:
            units.extend(_split_long_line(line, chunk_size))

    chunks: list[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer} {unit}".strip() if buffer else unit
        if len(candidate) <= chunk_size or not buffer:
            buffer = candidate
        else:
            chunks.append(buffer)
            tail = buffer[-chunk_overlap:].strip()
            buffer = f"{tail} {unit}".strip() if tail else unit
    if buffer:
        chunks.append(buffer)
    return chunks


def _tokenizer():
    """tiktoken-энкодер для оценки длины чанков в токенах (кэшируется)."""
    global _ENC
    try:
        return _ENC
    except NameError:
        pass
    import tiktoken

    _ENC = tiktoken.get_encoding(TOKENIZER_NAME)
    return _ENC


def _count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text))


def _split_semantic_atoms(text: str) -> list[str]:
    """Абзацы -> предложения: минимальные смысловые атомы для склейки в чанки."""
    atoms: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
            sentence = sentence.strip()
            if sentence:
                atoms.append(sentence)
    return atoms


def _split_oversized_atom(atom: str, max_tokens: int) -> list[str]:
    """Атом (например, длинная строка таблицы без знаков конца предложения)

    режется по словам на куски не длиннее max_tokens, чтобы ни один чанк
    не выходил за верхнюю границу.
    """
    if _count_tokens(atom) <= max_tokens:
        return [atom]
    pieces: list[str] = []
    buffer: list[str] = []
    for word in atom.split():
        buffer.append(word)
        if _count_tokens(" ".join(buffer)) >= max_tokens:
            pieces.append(" ".join(buffer))
            buffer = []
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def split_text_tokens(
    text: str,
    *,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> list[str]:
    """Чанкинг по смысловым границам: чанки ~target_tokens (не длиннее max),

    между соседними чанками сохраняется перекрытие ~overlap_ratio от target
    (по умолчанию 12%, что укладывается в требование 10..15%). Атомы — это
    предложения, поэтому границы проходят по границам предложений/абзацев.
    """
    atoms: list[str] = []
    for atom in _split_semantic_atoms(text):
        atoms.extend(_split_oversized_atom(atom, max_tokens))
    if not atoms:
        return []

    overlap_tokens = max(1, int(target_tokens * overlap_ratio))
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for atom in atoms:
        atom_tokens = _count_tokens(atom)
        if cur and cur_tokens + atom_tokens > max_tokens:
            chunks.append(" ".join(cur))
            # Перекрытие: хвостовые атомы предыдущего чанка в пределах бюджета.
            keep: list[str] = []
            keep_tokens = 0
            for prev in reversed(cur):
                prev_tokens = _count_tokens(prev)
                if keep_tokens + prev_tokens > overlap_tokens:
                    break
                keep.insert(0, prev)
                keep_tokens += prev_tokens
            # Если с перекрытием атом всё равно не влезает — перекрытие отбрасываем.
            if keep_tokens + atom_tokens > max_tokens:
                keep, keep_tokens = [], 0
            cur, cur_tokens = keep, keep_tokens
        cur.append(atom)
        cur_tokens += atom_tokens

    if cur:
        chunks.append(" ".join(cur))

    # Склеиваем «хвостики» меньше min_tokens с соседом, если это не ломает max.
    merged: list[str] = []
    for chunk in chunks:
        if (
            merged
            and _count_tokens(chunk) < min_tokens
            and _count_tokens(merged[-1]) + _count_tokens(chunk) <= max_tokens
        ):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def parse_pdf_to_chunks(
    path: Path,
    *,
    label: str,
    title: str,
    kind: str,
    ocr_pages: dict[int, str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    mode: str = CHUNKING_MODE,
) -> list[Chunk]:
    """PDF -> чанки. Каждый чанк привязан к одной странице (page/header/figure).

    mode="semantic" — по смысловым границам (300..500 токенов, перекрытие 10..15%);
    mode="chars"    — простой посимвольный чанкинг (legacy).
    """
    chunks: list[Chunk] = []
    for page in parse_pdf(
        path, label=label, title=title, kind=kind, ocr_pages=ocr_pages
    ):
        if mode == "semantic":
            parts = split_text_tokens(page.text)
        else:
            parts = split_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for part in parts:
            chunks.append(Chunk(text=part, metadata=dict(page.metadata)))
    return chunks


def load_all_chunks(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    mode: str = CHUNKING_MODE,
) -> list[Chunk]:
    """Все документы из манифеста -> список чанков с метаданными."""
    manifest = load_manifest()
    chunks: list[Chunk] = []
    for entry in manifest["files"]:
        path = PDF_DIR / entry["path"]
        if not path.exists():
            print(f"[parsing] пропущен отсутствующий файл: {path}")
            continue
        ocr = load_ocr(path)
        chunks.extend(
            parse_pdf_to_chunks(
                path,
                label=entry["label"],
                title=entry.get("title", entry["label"]),
                kind=entry.get("kind", "document"),
                ocr_pages=ocr,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                mode=mode,
            )
        )
    return chunks


def save_chunks(chunks: list[Chunk]) -> None:
    """Артефакт для отчёта: дамп чанков с метаданными в JSON."""
    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "text": chunk.text,
            "metadata": {key: (None if value is None else value) for key, value in chunk.metadata.items()},
        }
        for chunk in chunks
    ]
    CHUNKS_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )