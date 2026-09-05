"""Быстрый пересчёт метрики задачи 1 по сохранённым holdout-предсказаниям.

Обучение и инференс не запускаются: берутся файлы `reports/gapfill/improvement/validation_*.csv`,
где на каждую контрольную строку записаны истинное значение, прогноз модели (`prior`)
и прогноз после калибровки (`pred`). Этого достаточно, чтобы получить RMSE и GapScore
за доли секунды вместо часов переобучения.

Запуск:
    uv run --no-sync python -m gapfill.metrics                 # пересчитать и обновить отчёт
    uv run --no-sync python -m gapfill.metrics --no-write      # только показать числа
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gapfill.config import ROOT

IMPROVEMENT_DIR = ROOT / "reports" / "gapfill" / "improvement"
VALIDATION_PATH = ROOT / "reports" / "gapfill" / "validation.json"
TARGET = "primary_ndvi"

# Метрика организаторов: GapScore = round(30 · max(0, 1 − RMSE / 0.10), 2)
GAP_SCALE = 0.10
GAP_MAX = 30.0


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Корень из средней квадратичной ошибки одним проходом по массиву.

    Это та же метрика, что и `gapfill.data.rmse`, но без обращения к данным кейса: модуль должен
    считаться на голом окружении, где стоят только numpy и pandas.
    """
    error = prediction - truth
    return float(np.sqrt(np.dot(error, error) / error.size))


def gap_score(value: float) -> float:
    """Баллы автометрики по RMSE."""
    return round(GAP_MAX * max(0.0, 1.0 - value / GAP_SCALE), 2)


def read_holdout(directory: Path = IMPROVEMENT_DIR) -> dict[str, pd.DataFrame]:
    """Файлы holdout по сидам валидации. Читаются только нужные колонки."""
    columns = ["pid", "date", "split", TARGET, "prior", "pred"]
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("validation_*.csv")):
        seed = path.stem.split("_")[-1]
        frames[seed] = pd.read_csv(path, usecols=columns, dtype={"split": "category"})
    return frames


def score_frame(frame: pd.DataFrame) -> dict:
    """Ошибки для двух страт и двух вариантов прогноза.

    `all` — все контрольные строки holdout, `target` — только строки конкурсного файла (split == test).
    `model` — чистый ансамбль, `calibrated` — он же после калибровки историческими агрегатами.
    """
    truth = frame[TARGET].to_numpy(dtype=float)
    is_target = (frame["split"] == "test").to_numpy()
    result: dict[str, dict] = {}
    for strata, mask in (("all", np.ones(len(frame), dtype=bool)), ("target", is_target)):
        block: dict[str, dict] = {"n": int(mask.sum())}
        for label, column in (("model", "prior"), ("calibrated", "pred")):
            value = rmse(truth[mask], frame[column].to_numpy(dtype=float)[mask])
            block[label] = {"rmse": round(value, 4), "gap_score": gap_score(value)}
        result[strata] = block
    return result


def aggregate(scores: dict[str, dict]) -> dict:
    """Среднее и разброс по сидам валидации: один сид — ещё не результат."""
    summary: dict[str, dict] = {}
    for strata in ("all", "target"):
        block: dict[str, dict] = {}
        for label in ("model", "calibrated"):
            values = [scores[seed][strata][label]["rmse"] for seed in scores]
            mean = float(np.mean(values))
            block[label] = {
                "rmse": round(mean, 4),
                "rmse_spread": round(float(np.max(values) - np.min(values)), 4),
                "gap_score": gap_score(mean),
                "seeds": len(values),
            }
        summary[strata] = block
    return summary


def build_report(directory: Path = IMPROVEMENT_DIR) -> dict:
    """Полный отчёт: по каждому сиду и в среднем."""
    frames = read_holdout(directory)
    if not frames:
        raise FileNotFoundError(f"нет файлов validation_*.csv в {directory}")
    scores = {seed: score_frame(frame) for seed, frame in frames.items()}
    return {"by_seed": scores, "mean": aggregate(scores), "n_rows": int(sum(len(f) for f in frames.values()))}


def merge_into_validation(report: dict, path: Path = VALIDATION_PATH) -> dict:
    """Обновляет reports/gapfill/validation.json, сохраняя схему, которую читает сервис.

    Заголовочная цифра — калиброванная модель на страте конкурсного файла: именно она уходит
    в submission. Чистый ансамбль остаётся рядом, чтобы был виден вклад калибровки.
    """
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    mean = report["mean"]
    current.update({
        "rmse": mean["target"]["calibrated"]["rmse"],
        "rmse_testlike": mean["all"]["calibrated"]["rmse"],
        "gap_score": mean["target"]["calibrated"]["gap_score"],
        "rmse_model_only": mean["target"]["model"]["rmse"],
        "gap_score_model_only": mean["target"]["model"]["gap_score"],
        "rmse_spread": mean["target"]["calibrated"]["rmse_spread"],
        "n_val_points": report["n_rows"] // max(1, len(report["by_seed"])),
        "val_seeds": sorted(report["by_seed"]),
        "models": "LightGBM (kriging-признаки, 2 seeds) + остаточная SeasonNet, затем калибровка "
                  "историческими агрегатами",
        "source": "reports/gapfill/improvement/validation_*.csv",
        "status": "local_holdout_not_private_leaderboard",
    })
    path.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
    return current


def main() -> None:
    """Пересчёт метрики и обновление отчёта."""
    parser = argparse.ArgumentParser(description="Пересчёт RMSE и GapScore по holdout-предсказаниям")
    parser.add_argument("--dir", default=str(IMPROVEMENT_DIR), help="каталог с validation_*.csv")
    parser.add_argument("--no-write", action="store_true", help="не обновлять validation.json")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(Path(args.dir))
    elapsed = time.perf_counter() - started

    mean = report["mean"]
    print(f"Сидов валидации: {len(report['by_seed'])}, строк: {report['n_rows']}, время {elapsed:.2f} с")
    for strata in ("target", "all"):
        for label in ("model", "calibrated"):
            block = mean[strata][label]
            print(f"  {strata:6s} {label:10s} RMSE {block['rmse']:.4f} "
                  f"(разброс {block['rmse_spread']:.4f})  GapScore {block['gap_score']:.2f}")

    if not args.no_write:
        merge_into_validation(report)
        print(f"Обновлён {VALIDATION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
