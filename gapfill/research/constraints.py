"""Проверка и использование опубликованных исторических агрегатов как ограничений.

Агрегаты могут содержать информацию о скрытых значениях. Это отдельная калибровка по входным
файлам конкурса, а не улучшение способности модели обобщать на новые территории.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

from gapfill.config import ARTIFACTS_DIR, EXTRA_PATHS, TARGET, TRAIN_PATH
from gapfill.data import load_all, make_mask


def published_norms(held=None):
    """Берёт только действительно опубликованные нормы видимых строк старых файлов."""
    parts = []
    for path in (TRAIN_PATH, *EXTRA_PATHS):
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame.date)
        frame["pid"] = frame.anon_polygon_id
        if "ndvi_climatology_mean" in frame:
            parts.append(frame.loc[frame.ndvi_climatology_mean.notna()])
    norms = pd.concat(parts, ignore_index=True).drop_duplicates(["pid", "date"])
    if held is not None:
        keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
        norms = norms.loc[~pd.MultiIndex.from_frame(norms[["pid", "date"]]).isin(keys)]
    return norms


def system(rows, norms, unknown, window=8):
    """A x = b для сумм и A x² = c для сумм квадратов из mean/std ddof=0."""
    days = rows.doy.to_numpy()
    years = rows.year.to_numpy()
    ndays = norms.date.dt.dayofyear.to_numpy()
    nyears = norms.date.dt.year.to_numpy()
    mask = (np.abs(ndays[:, None] - days[None, :]) <= window) & (nyears[:, None] != years[None, :])
    count = mask.sum(1)
    known_y = np.where(unknown, 0., rows[TARGET].fillna(0.).to_numpy())
    mean = norms.ndvi_climatology_mean.to_numpy()
    std = norms.ndvi_climatology_std.to_numpy()
    b = count * mean - mask @ known_y
    c = count * (mean ** 2 + std ** 2) - mask @ (known_y ** 2)
    return mask[:, unknown].astype(float), b, c, count


def diagnostic(obs, grid, held=None):
    """Проверяет точность формулы там, где все её слагаемые известны."""
    held = obs.iloc[:0] if held is None else held
    norms = published_norms(held)
    rows = grid.loc[grid[TARGET].notna() | grid.is_gap].copy()
    unknown_keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
    rows["unknown"] = rows.is_gap | pd.MultiIndex.from_frame(rows[["pid", "date"]]).isin(unknown_keys)
    report, checked = {}, []
    for pid, group in rows.groupby("pid"):
        ng = norms.loc[norms.pid.eq(pid)]
        a, b, _c, count = system(group, ng, group.unknown.to_numpy())
        exact = a.sum(1) == 0
        if exact.any():
            checked.extend((b[exact] / count[exact]).tolist())
        useful = a.sum(1) > 0
        singular = np.linalg.svd(a[useful], compute_uv=False) if useful.any() else np.array([])
        report[pid] = {"unknown": int(group.unknown.sum()), "equations": int(useful.sum()),
                       "rank": int((singular > 1e-8).sum()),
                       "known_equations": int(exact.sum()),
                       "max_known_mean_error": float(np.max(np.abs(b[exact] / count[exact]))) if exact.any() else None}
    return {"known_equations": len(checked), "max_known_error": float(np.max(np.abs(checked))) if checked else None,
            "rmse_known_error": float(np.sqrt(np.mean(np.array(checked) ** 2))) if checked else None,
            "polygons": report}


def linear_projection(prior, a, b, strength=1., ridge=0., scale=None):
    """Минимальная поправка к прогнозу, согласующая его с известными суммами."""
    delta = b - a @ prior
    if scale is not None:
        transformed = a * scale[None, :]
        return prior + strength * scale * lstsq(transformed, delta, cond=1e-9, lapack_driver="gelsd")[0]
    if ridge:
        return prior + strength * a.T @ np.linalg.solve(a @ a.T + ridge * np.eye(len(a)), delta)
    return prior + strength * lstsq(a, delta, cond=1e-9, lapack_driver="gelsd")[0]


def main():
    """Сохраняет проверку точности опубликованных агрегатов без обучения модели."""
    p = argparse.ArgumentParser()
    p.add_argument("--val-seed", type=int)
    args = p.parse_args()
    obs, grid, _ = load_all()
    held = obs.loc[make_mask(obs, seed=args.val_seed)] if args.val_seed is not None else None
    report = diagnostic(obs, grid, held)
    out = ARTIFACTS_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"constraints_audit_{args.val_seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
