"""Ссылки на исходные документы для ответов API и веб-чата.

Строит карту «метка источника -> файл» из манифеста (``data/raw/manifest.json``),
чтобы ответы содержали кликабельные ссылки на документы, которые реально искали
(источники в ``POST /answer`` и цитаты в ответе).

Отдаются только файлы, перечисленные в манифесте: FastAPI обслуживает их через
``GET /sources/{name}`` (см. :mod:`normdoc.api`), поэтому произвольные пути с
диска прочитать нельзя.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from normdoc.config import MANIFEST_PATH, PDF_DIR

#: Префикс публичного маршрута, по которому FastAPI отдаёт исходники.
SOURCE_ROUTE = "/sources"

_MAP: dict[str, Path] | None = None


def _build_map() -> dict[str, Path]:
    """Читает манифест и строит {метка: абсолютный путь к PDF} внутри ``PDF_DIR``."""
    mapping: dict[str, Path] = {}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return mapping

    root = PDF_DIR.resolve()
    for entry in manifest.get("files", []):
        rel = entry.get("path")
        label = entry.get("label") or entry.get("source")
        if not rel or not label:
            continue
        path = (PDF_DIR / rel).resolve()
        # Защита от выхода за пределы каталога подготовленных PDF.
        if not path.is_relative_to(root):
            continue
        mapping[str(label)] = path
    return mapping


def source_map() -> dict[str, Path]:
    """Кэшированная карта меток источников на файлы документов."""
    global _MAP
    if _MAP is None:
        _MAP = _build_map()
    return _MAP


def resolve_source(name: str) -> Path | None:
    """Находит документ по метке источника (или по имени файла).

    Возвращает ``None``, если источника нет в манифесте или файл отсутствует.
    """
    if not name:
        return None
    path = source_map().get(name)
    if path is not None:
        return path if path.is_file() else None
    for candidate in source_map().values():
        if candidate.name == name:
            return candidate if candidate.is_file() else None
    return None


def source_url(label: str | None, page: int | None = None) -> str | None:
    """Кликабельная ссылка на документ источника (и нужную страницу), либо ``None``.

    Ссылка относительная; фрагмент ``#page=N`` открывает PDF сразу на странице
    (поддерживается встроенными просмотрщиками PDF в браузерах).
    """
    if not label:
        return None
    path = resolve_source(str(label))
    if path is None:
        return None
    url = f"{SOURCE_ROUTE}/{quote(path.name)}"
    if page is not None:
        url += f"#page={page}"
    return url
