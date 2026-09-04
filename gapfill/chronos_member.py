"""Zero-shot Chronos-2 как участник смеси: прогноз вперёд по левому контексту и назад по правому.

Запуск: uv run python -m gapfill.chronos_member --val-seed 777 --out chronos_v1
Ряд полигона-года — ежедневный гармонизированный NDVI (NaN там, где наблюдений нет); Chronos-2 маскирует
NaN в контексте. Предсказание = среднее прямого и обратного прогноза (взвешенное по расстоянию до ближайшего
известного дня) плюс смещение сенсора по правилу «сенсор по дате».
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from gapfill.config import ARTIFACTS_DIR, SENSOR_CODE, SENSOR_OFFSET, TARGET
from gapfill.data import load_all, make_mask, polygon_kinds
from gapfill.dataset import val_examples
from gapfill.nn_data import SEASON_LEN, build_tensors
from gapfill.nn_model import query_days_from_obs, val_metrics
from gapfill.features_series import harmonize

MIN_CONTEXT = 3       # минимум известных точек в контексте, иначе сторона не используется
BATCH = 256


def _forecast(pipeline, contexts: list[np.ndarray]) -> np.ndarray:
    """Медианный прогноз на один шаг вперёд для списка контекстов (NaN допустимы)."""
    out = np.full(len(contexts), np.nan, np.float32)
    for i in range(0, len(contexts), BATCH):
        chunk = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i + BATCH]]
        preds = pipeline.predict(chunk, prediction_length=1)
        for j, p in enumerate(preds):
            arr = p.float().numpy() if hasattr(p, "numpy") else np.asarray(p, dtype=np.float32)
            out[i + j] = float(np.nanmedian(arr.reshape(-1)))
    return out


def _sides(h: np.ndarray, known: np.ndarray, sample: np.ndarray, day: np.ndarray) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Левые и правые контексты для запросов (sample, day); расстояния до ближайшего известного дня."""
    left, right, d_left, d_right = [], [], np.full(len(day), np.inf), np.full(len(day), np.inf)
    for k, (s, d) in enumerate(zip(sample, day)):
        series = np.where(known[s], h[s], np.nan)
        lctx, rctx = series[:d].copy(), series[d + 1:][::-1].copy()
        kl, kr = np.flatnonzero(~np.isnan(lctx)), np.flatnonzero(~np.isnan(rctx))
        left.append(lctx if len(kl) >= MIN_CONTEXT else None)
        right.append(rctx if len(kr) >= MIN_CONTEXT else None)
        if len(kl):
            d_left[k] = d - kl[-1]
        if len(kr):
            d_right[k] = kr[0] + 1
    return left, right, d_left, d_right


def predict_points(pipeline, h, known, sample, day) -> np.ndarray:
    """Комбинированный прогноз Chronos-2 для точек (sample, day) в шкале S2."""
    left, right, dl, dr = _sides(h, known, sample, day)
    fwd = np.full(len(day), np.nan, np.float32)
    bwd = np.full(len(day), np.nan, np.float32)
    idx = [i for i, c in enumerate(left) if c is not None]
    if idx:
        fwd[idx] = _forecast(pipeline, [left[i] for i in idx])
    idx = [i for i, c in enumerate(right) if c is not None]
    if idx:
        bwd[idx] = _forecast(pipeline, [right[i] for i in idx])
    wl = np.where(np.isnan(fwd), 0.0, 1.0 / np.maximum(dl, 1))
    wr = np.where(np.isnan(bwd), 0.0, 1.0 / np.maximum(dr, 1))
    num = np.nan_to_num(fwd) * wl + np.nan_to_num(bwd) * wr
    return np.where(wl + wr > 0, num / np.maximum(wl + wr, 1e-9), np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos-2 zero-shot для валидационных точек")
    parser.add_argument("--val-seed", type=int, default=777)
    parser.add_argument("--model", type=str, default="amazon/chronos-2")
    parser.add_argument("--out", type=str, default="chronos_v1")
    args = parser.parse_args()
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda" if torch.cuda.is_available() else "cpu")
    obs, grid, _ = load_all()
    meta = obs.assign(poly_kind=obs["pid"].map(polygon_kinds(obs)).astype(str), is_2025=obs["year"].eq(2025))
    t = build_tensors(obs, grid)
    val_mask = make_mask(obs, seed=args.val_seed)
    known = t.known.copy()
    ok = t.idx >= 0
    known[ok] &= ~val_mask[t.idx[ok]]
    offsets = np.array([SENSOR_OFFSET[s] for s in SENSOR_CODE], np.float32)
    h = t.y - offsets[np.maximum(t.sensor, 0)]
    q = query_days_from_obs(t, val_mask)
    sample, day = np.nonzero(q)
    pred_h = predict_points(pipeline, h, known, sample, day)
    pred_days = np.full(t.y.shape, np.nan, np.float32)
    pred_days[sample, day] = pred_h
    # смещение сенсора по правилу из табличных признаков (rule_sensor), без него — 0
    X_val, meta_val = val_examples(obs, grid, args.val_seed)
    rule = X_val["rule_sensor"].to_numpy()
    rule_off = np.where(rule >= 0, offsets[np.clip(rule, 0, 2).astype(int)], 0.0)
    pred_obs = np.full(len(obs), np.nan, np.float32)
    pred_obs[t.idx[ok]] = pred_days[ok]
    val_rows = np.flatnonzero(val_mask)
    fill = X_val["nb_interp_val"].to_numpy()
    pred = np.where(np.isnan(pred_obs[val_rows]), fill, pred_obs[val_rows] + rule_off)
    metrics, _ = val_metrics(np.where(np.isnan(pred_days), 0, pred_days), t, meta, val_mask)
    out_dir = ARTIFACTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_val.assign(pred=pred).to_parquet(out_dir / "val_pred.parquet")
    y = meta_val[TARGET].to_numpy()
    result = {"rmse_with_rule_offset": float(np.sqrt(np.nanmean((pred - y) ** 2))),
              "share_no_forecast": float(np.isnan(pred_obs[val_rows]).mean()), "raw_metrics_no_offset": metrics}
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
