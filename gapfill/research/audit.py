"""Аудит данных перед экспериментами; это анализ выборки, а не набор тестов."""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR, INDEX_COLS, ROOT, TARGET
from gapfill.data import load_all, make_mask


def main():
    """Сохраняет состав выборки, пересечения файлов и доступность дополнительных сенсоров."""
    obs, grid, gaps = load_all()
    report = {"n_obs": len(obs), "n_gaps": len(gaps), "cpu_count": os.cpu_count()}
    report["files"] = {}
    for path in [*sorted((ROOT / "data").glob("*.csv")), ROOT / "test_features (1) (2).csv"]:
        frame = pd.read_csv(path)
        report["files"][path.name] = {
            "rows": len(frame), "polygons": frame.anon_polygon_id.nunique(),
            "known": int(frame.primary_ndvi.notna().sum()),
            "sha256_lf": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        }
    report["observations_by_split"] = obs.groupby("split").size().to_dict()
    report["sensor_values"] = obs[INDEX_COLS].count().to_dict()
    report["primary_sensor"] = obs.groupby("sensor").size().to_dict()
    report["target_quantiles"] = obs[TARGET].quantile([0, .001, .01, .5, .99, .999, 1]).to_dict()
    report["exact_duplicate_values"] = {}
    report["correlated_fields"] = {}
    for sensor in ("s2", "landsat", "modis"):
        piv = obs.pivot(index="day_num", columns="pid", values=f"{sensor}_ndvi")
        corr = piv.corr(min_periods=50)
        corr = corr.mask(np.eye(len(corr), dtype=bool))
        pairs = corr.stack().sort_values(ascending=False).head(12)
        report["correlated_fields"][sensor] = {str(k): float(v) for k, v in pairs.items()}
        z = obs.loc[obs[f"{sensor}_ndvi"].notna()]
        report["exact_duplicate_values"][sensor] = int(z.duplicated(["day_num", f"{sensor}_ndvi"]).sum())
    for seed in (777, 999, 2026):
        mask = make_mask(obs, seed=seed)
        val = obs.loc[mask]
        report[f"validation_{seed}"] = {
            "rows": len(val), "target_file_rows": int(val.split.eq("test").sum()),
            "target_file_outside_0_1": int((val.split.eq("test") & ~val[TARGET].between(0, 1)).sum()),
        }
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "mps": torch.backends.mps.is_available(),
                           "cuda": torch.cuda.is_available()}
    except ImportError as exc:
        report["torch"] = str(exc)
    out = ARTIFACTS_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
