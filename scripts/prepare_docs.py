"""Подготовка исходных документов -> data/raw/pdf + data/raw/ocr + manifest.json.

Шаг выполняется на хосте (нужны libreoffice и tesseract), результат коммитится,
поэтому запуск в контейнере не требуется.

1. Конвертация RTF/ODT в PDF (soffice --headless).
2. OCR сканов без текстового слоя (pdftoppm + tesseract -l rus) в data/raw/ocr/*.json.
3. Дедупликация по текстовому содержимому (одинаковые ГОСТы оставляем один).
4. Сборка data/raw/manifest.json — список файлов, меток и названий.

Запуск:
  uv run python scripts/prepare_docs.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from normdoc.config import OCR_DIR, PDF_DIR, RAW_DIR, SOURCES_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Описание корпуса: файл -> (label, title, kind)
# label используем в цитатах ([label, стр. N]).
# ---------------------------------------------------------------------------
DOCS: list[dict] = [
    {
        "file": "GOST_2019_PO.pdf",
        "label": "ГОСТ Р 57100—2016",
        "title": "ГОСТ Р 57100—2016 (ISO/IEC/IEEE 42010:2011) Системная и программная инженерия. Описание архитектуры",
        "kind": "standard",
    },
    {
        "file": "4293751719.pdf",
        "label": "ГОСТ Р 57100—2016 (копия)",
        "title": "Копия ГОСТ Р 57100—2016 (исключается дедупликацией)",
        "kind": "duplicate",
    },
    {
        "file": "_ГОСТ Р 57100-2025. Национальный стандарт Российской Федерации. Системная и программная инженерия. Описание архитектуры_ (утв. и введен в дейс.pdf",
        "label": "ГОСТ Р 57100—2025",
        "title": "ГОСТ Р 57100-2025 (введён приказом Росстандарта от 25.03.2025 N 215-ст) Системная и программная инженерия. Описание архитектуры",
        "kind": "standard",
    },
    {
        "file": "4293729971.pdf",
        "label": "ГОСТ Р 58412—2019",
        "title": "ГОСТ Р 58412-2019 Защита информации. Разработка безопасного программного обеспечения. Угрозы безопасности информации при разработке программного обеспечения",
        "kind": "standard",
    },
    {
        "file": "51439.pdf",
        "label": "ГОСТ Р 54136—2010",
        "title": "ГОСТ Р 54136-2010 Системы промышленной автоматизации и интеграция. Руководство по применению стандартов, структура и словарь",
        "kind": "standard",
    },
    {
        "file": "75194.pdf",
        "label": "ГОСТ Р 59407—2021",
        "title": "ГОСТ Р 59407-2021 Информационные технологии. Методы и средства обеспечения безопасности. Базовая архитектура защиты персональных данных",
        "kind": "standard",
    },
    {
        "file": "83777.pdf",
        "label": "ГОСТ Р 56939—2024",
        "title": "ГОСТ Р 56939-2024 Защита информации. Разработка безопасного программного обеспечения. Общие требования",
        "kind": "standard",
    },
    {
        "file": "Приказ ФСТЭК РФ от 01.12.2023 N 240 Об утверждении Порядка проведения сертификации процессов.rtf",
        "label": "Приказ ФСТЭК России от 01.12.2023 №240",
        "title": "Приказ ФСТЭК России от 01.12.2023 №240 «Порядок проведения сертификации процессов безопасной разработки ПО средств защиты информации» (в ред. от 30.06.2025 №230)",
        "kind": "law",
    },
    {
        "file": "Заявка на сертификацию процессов безопасной разработки программного обеспечения (приложение N 1 к.rtf",
        "label": "Заявка на сертификацию (приложение №1 к приказу №240)",
        "title": "Заявка на сертификацию процессов безопасной разработки программного обеспечения (рекомендуемый образец, приложение №1 к приказу ФСТЭК №240)",
        "kind": "form",
    },
    {
        "file": "Требования по безопасности информации. Утверждены приказом ФСТЭК России от 4 июля 2022 г. N 118.odt",
        "label": "Требования ФСТЭК №118 (контейнеризация)",
        "title": "Требования по безопасности информации к средствам контейнеризации (приказ ФСТЭК России от 04.07.2022 №118, выписка)",
        "kind": "law",
    },
    {
        "file": "Распоряжение Правительства РФ от 28.04.2018 N 824-р.pdf",
        "label": "Распоряжение Правительства РФ от 28.04.2018 №824-р",
        "title": "Распоряжение Правительства РФ от 28.04.2018 №824-р (о закупках у единственного поставщика, малые закупки ЕАТ)",
        "kind": "law",
    },
    {
        "file": "Регламент_версия 19_V2.pdf",
        "label": "Регламент ЕАТ (редакция №19)",
        "title": "Регламент функционирования Единого агрегатора торговли (редакция №19)",
        "kind": "regulation",
    },
    {
        "file": "Письмо Федерального казначейства от 27.02.2019 №07-04-05 14-1374.pdf",
        "label": "Письмо Казначейства от 27.02.2019 №07-04-05/14-1374",
        "title": "Письмо Федерального казначейства от 27.02.2019 №07-04-05/14-1374 (разъяснения по работе ЕАТ)",
        "kind": "law",
    },
    {
        "file": "Краткая_информация_по_обновлению_Версия_2_80_1 (2).pdf",
        "label": "Обновление ЕАТ (версия 2.80.1)",
        "title": "Краткая информация по обновлению ЕАТ, версия 2.80.1",
        "kind": "release_notes",
    },
]

PDF_LABEL = "pdf"


def _slug(label: str) -> str:
    keep = "".join(c for c in label if c.isalnum() or c in " _-«»—")
    return keep.strip().replace(" ", "_")


def _convert_to_pdf(src: Path, out_dir: Path) -> Path | None:
    """libreoffice: RTF/ODT -> PDF (с ретраями на случай нехватки памяти)."""
    tmp = out_dir / "_conv"
    tmp.mkdir(parents=True, exist_ok=True)
    # Важно: чистим временный каталог перед конвертацией, иначе candidates[0]
    # может оказаться PDF от предыдущего файла (одинаковые документы и
    # ложная дедупликация).
    for stale in tmp.glob("*.pdf"):
        stale.unlink(missing_ok=True)
    import time as _t

    last_err = ""
    for attempt in range(1, 4):
        res = subprocess.run(
            [
                "soffice",
                "--headless",
                "-env:UserInstallation=file:///tmp/lo_profile",
                "--convert-to",
                "pdf:writer_pdf_Export",
                "--outdir",
                str(tmp),
                str(src),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if res.returncode == 0:
            break
        last_err = f"{res.returncode}: {res.stderr[-400:]}"
        print(f"[prepare] повтор конвертации {src.name} (попытка {attempt})")
        _t.sleep(10)
    else:
        print(f"[prepare] ошибка конвертации {src.name}: {last_err.strip()}")
        return None
    candidates = list(tmp.glob("*.pdf"))
    if not candidates:
        return None
    # Копируем результат в стабильный путь out_dir/<имя>.pdf, чтобы:
    #  * следующий запуск нашёл готовый PDF (_preconvert.s existing);
    #  * очистка временного каталога не удаляла результат прошлой конвертации.
    stable = out_dir / f"{src.stem}.pdf"
    shutil.copy2(candidates[0], stable)
    return stable


def _text_hash(pdf: Path) -> str | None:
    try:
        import pymupdf

        with pymupdf.open(pdf) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        if not text.strip():
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except Exception as exc:  # pragma: no cover
        print(f"[prepare] не удалось прочитать {pdf.name}: {exc}")
        return None


def _needs_ocr(pdf: Path, full_text: str) -> bool:
    pages = len(full_text.split("\n\n"))
    nonempty = sum(1 for block in full_text.split("\n") if block.strip())
    total_pages = _page_count(pdf)
    # Считаем, что скан: менее 30% страниц имеют текст.
    if total_pages == 0:
        return False
    text_pages = 0
    try:
        import pymupdf

        with pymupdf.open(pdf) as doc:
            for page in doc:
                if page.get_text("text").strip():
                    text_pages += 1
    except Exception:
        text_pages = nonempty
    return text_pages / total_pages < 0.3 and total_pages > 0


def _page_count(pdf: Path) -> int:
    try:
        import pymupdf

        with pymupdf.open(pdf) as doc:
            return doc.page_count
    except Exception:
        return 0


def _ocr_pdf(pdf: Path, stub: str) -> None:
    """Рендер страниц и распознавание текста tesseract'ом -> sidecar JSON."""
    work = OCR_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []
    try:
        import pymupdf

        with pymupdf.open(pdf) as doc:
            total = doc.page_count
            for i in range(total):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=300)
                img = work / f"p{i + 1:04}.png"
                pix.save(str(img))
                res = subprocess.run(
                    ["tesseract", str(img), "stdout", "-l", "rus", "--dpi", "300"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                pages.append({"physical": i + 1, "text": res.stdout.strip()})
                img.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[prepare] OCR-ошибка для {pdf.name}: {exc}")
        return
    side = OCR_DIR / f"{stub}.json"
    side.write_text(json.dumps({"source": pdf.name, "pages": pages}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[prepare] OCR готов: {side} ({len(pages)} стр.)")


def _preconvert() -> dict[str, Path]:
    """Фаза 1: конвертируем ВСЕ RTF/ODT заранее (до анализа PDF), чтобы
    не конкурировать за память с PyMuPDF/tesseract (soffice падает с bad_alloc,
    если память уже потрачена). Возвращает {имя_файла: pdf_путь}."""
    conv_dir = PDF_DIR / "_conv"
    conv_dir.mkdir(parents=True, exist_ok=True)
    converted: dict[str, Path] = {}
    for doc in DOCS:
        src = SOURCES_DIR / doc["file"]
        if src.suffix.lower() in {".pdf"}:
            continue
        # Ищем уже сконвертированный файл.
        existing = conv_dir / f"{src.stem}.pdf"
        if existing.exists():
            converted[doc["file"]] = existing
            continue
        pdf = _convert_to_pdf(src, conv_dir)
        if pdf is None:
            continue
        converted[doc["file"]] = pdf
    return converted


def build_manifest(force: bool) -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)

    converted = _preconvert()

    manifest_files: list[dict] = []
    deduplicated: list[dict] = []
    seen_hashes: dict[str, str] = {}

    for doc in DOCS:
        src = SOURCES_DIR / doc["file"]
        if not src.exists():
            print(f"[prepare] пропуск отсутствующего файла: {src.name}")
            continue

        if src.suffix.lower() == ".pdf":
            pdf_cand = src
        else:
            if doc["file"] not in converted:
                print(f"[prepare] конвертация {src.name} не удалась, файл пропущен")
                continue
            pdf_cand = converted[doc["file"]]

        label = doc["label"]
        stub = _slug(label)
        target = PDF_DIR / f"{stub}.pdf"

        if force or not target.exists():
            if pdf_cand != target:
                shutil.copy2(pdf_cand, target)
                print(f"[prepare] {src.name} -> {target.name}")

        # Дедупликация по тексту.
        h = _text_hash(target)
        if h is None:
            h = hashlib.sha256(target.name.encode()).hexdigest()
        if h in seen_hashes and seen_hashes[h] != label:
            deduplicated.append({"path": src.name, "duplicate_of": seen_hashes[h]})
            print(f"[prepare] дубль пропущен: {src.name} (совпадает с {seen_hashes[h]})")
            target.unlink(missing_ok=True)
            continue
        seen_hashes[h] = label

        # OCR сканов.
        full_text = _text_hash(target) or ""
        if _needs_ocr(target, full_text):
            side = OCR_DIR / f"{stub}.json"
            if force or not side.exists():
                print(f"[prepare] скан без текстового слоя: {target.name}, запускаю OCR...")
            _ocr_pdf(target, stub)

        manifest_files.append(
            {
                "path": f"{target.name}",
                "source": src.name,
                "label": label,
                "title": doc["title"],
                "kind": doc["kind"],
                "pages": _page_count(target),
            }
        )

    manifest_files.sort(key=lambda e: e["label"])
    manifest = {
        "_comment": "Подготовлено scripts/prepare_docs.py. path — файл относительно data/raw/pdf; label — короткая метка для цитат.",
        "files": manifest_files,
        "deduplicated": deduplicated,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / "manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] манифест: {p} ({len(manifest_files)} док., {len(deduplicated)} дублей)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Переделать конвертацию/OCR заново")
    args = parser.parse_args()
    return build_manifest(args.force)


if __name__ == "__main__":
    raise SystemExit(main())