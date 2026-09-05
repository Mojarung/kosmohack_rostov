"""Сводка о решении для интерфейса: метрики обеих задач, состав данных и источники.

Числа не зашиты в код, а читаются из артефактов, которые лежат в репозитории:
`reports/gapfill/validation.json` (валидация задачи 1), `reports/anomalies/summary.json`
и `reports/anomalies/quality.json` (детекция аномалий). Если файла нет, соответствующий блок
заполняется значениями по умолчанию, а сервис продолжает работать.
"""

from __future__ import annotations

import json
from pathlib import Path

from anomaly.config import REPORT_DIR
from gapfill.config import EXTRA_PATHS, ROOT, TEST_PATH, TRAIN_PATH

VALIDATION_PATH = ROOT / "reports" / "gapfill" / "validation.json"

SOURCES = [
    {"name": "Sentinel-2 L2A", "detail": "Earth Search STAC, маска облаков по SCL, 20 м"},
    {"name": "Landsat 8/9 Collection 2 L2", "detail": "Microsoft Planetary Computer, маска qa_pixel, 30 м"},
    {"name": "MODIS MOD13Q1 v061", "detail": "Planetary Computer, 16-дневный композит NDVI, 250 м"},
    {"name": "ERA5 (Open-Meteo)", "detail": "суточная температура и осадки по центроиду поля"},
    {"name": "OpenStreetMap", "detail": "готовые контуры landuse=farmland через Overpass API, подложка карты"},
    {"name": "Данные кейса", "detail": "train_dataset.csv и private_features.csv организаторов"},
]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _count_rows(path: Path) -> int:
    """Число строк CSV без заголовка (быстро, без разбора содержимого)."""
    try:
        with path.open("rb") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def build_meta(n_gaps: int) -> dict:
    """Собирает ответ /api/meta. n_gaps приходит из загруженных данных (число контрольных точек)."""
    validation = _read_json(VALIDATION_PATH)
    summary = _read_json(REPORT_DIR / "summary.json")
    quality = _read_json(REPORT_DIR / "quality.json")
    baselines = validation.get("baselines", {}).get("neighbors_mean", {})

    return {
        "task1": {
            "n_gaps": n_gaps,
            "rmse_val": validation.get("rmse", 0.0),
            "rmse_testlike": validation.get("rmse_testlike", 0.0),
            "gap_score": validation.get("gap_score", 0.0),
            "baseline_rmse": baselines.get("rmse", 0.093),
            "baseline_gap_score": baselines.get("gap_score", 2.1),
            "models": validation.get("models", ""),
            "train_points": validation.get("train_points", 0),
            # чистый ансамбль без калибровки историческими агрегатами: видно вклад калибровки
            "rmse_model_only": validation.get("rmse_model_only", 0.0),
            "gap_score_model_only": validation.get("gap_score_model_only", 0.0),
            "rmse_spread": validation.get("rmse_spread", 0.0),
            "val_seeds": len(validation.get("val_seeds", [])),
            "status": validation.get("status", ""),
        },
        "task2": {
            "n_polygons": summary.get("n_polygons", 0),
            "n_seasons": summary.get("n_seasons", 0),
            "n_episodes": summary.get("n_episodes", 0),
            "seasons_with_episode": quality.get("seasons_with_episode", 0.0),
            "precision_points": quality.get("precision_points_in_episodes", 0.0),
            "recall_critical": quality.get("recall_critical_points", 0.0),
            "by_cause": summary.get("episodes_by_cause", {}),
        },
        "data": {
            "test_file": TEST_PATH.name,
            "test_rows": _count_rows(TEST_PATH),
            "train_rows": _count_rows(TRAIN_PATH),
            "extra_file": next((p.name for p in EXTRA_PATHS if p.exists()), None),
        },
        "sources": SOURCES,
    }
