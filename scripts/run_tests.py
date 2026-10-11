#!/usr/bin/env python
"""Запуск тестов проекта (pytest) с сохранением отчёта в results/.

    uv run python scripts/run_tests.py
    uv run python scripts/run_tests.py -k TestCitations -v
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from normdoc.config import PROJECT_ROOT, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytest_args", nargs="*", help="Аргументы, пробрасываемые в pytest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = RESULTS_DIR / "pytest.txt"

    cmd = [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-q", *args.pytest_args]
    print("[run_tests]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr
    print(output)
    report.write_text(output, encoding="utf-8")
    print(f"[run_tests] отчёт: {report}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
