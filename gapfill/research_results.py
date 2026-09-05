"""Компактный перенос проверяемых результатов из рабочего кэша в отчёты проекта."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR, ROOT, TARGET
from gapfill.data import gap_score, rmse


def summarize(frame, column):
    """Ошибки всех контрольных строк и точного среза текущего конкурсного файла."""
    result = {}
    for label, selection in {"all": frame.is_val, "target": frame.is_val & frame.split.eq("test")}.items():
        sample = frame.loc[selection]
        value = rmse(sample[TARGET], sample[column])
        result[label] = {"n": len(sample), "rmse": value, "gap_score": gap_score(value)}
    return result


def main():
    """Сохраняет ответы и предсказания holdout, достаточные для пересчёта таблицы RMSE."""
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    out = ROOT / "reports" / "gapfill" / "improvement"
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.runs:
        directory = ARTIFACTS_DIR / "research" / name
        result = json.loads((directory / "result.json").read_text())
        seed = result["args"]["val_seed"]
        prediction_source = directory / "pred.parquet"
        if not prediction_source.exists():
            prediction_source = directory / "blend_0.6.parquet"
        frame = pd.read_parquet(prediction_source)
        prediction_path = out / f"validation_{seed}.csv"
        columns = ["pid", "date", "split", TARGET, "prior", "pred"]
        frame.loc[frame.is_val, columns].to_csv(prediction_path, index=False, encoding="utf-8")
        source = result
        if "components" in result["args"]:
            source = json.loads((Path(result["args"]["components"]).parent / "result.json").read_text())
        reports.append({"run": name, "val_seed": seed, "args": result["args"], "model_args": source["args"],
                        "calibration_method": result["args"].get("calibration", "posterior"),
                        "model_only": summarize(frame, "prior"), "calibrated": summarize(frame, "pred"),
                        "predictions": prediction_path.name,
                        "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    report = {"status": "local_holdout_not_private_leaderboard", "source_commit":
              "2f6b2a4c8a1d10d945d44a3bc826e4d652c3e986", "runs": reports,
              "warning": "Калибровка использует опубликованные агрегаты, содержащие скрытые значения."}
    (out / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
