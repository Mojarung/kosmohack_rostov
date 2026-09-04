"""Пути проекта в одном месте, чтобы скрипты не зависели от рабочей директории."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "cache"

for _d in (ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)
