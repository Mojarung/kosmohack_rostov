"""Единый пайплайн «ряд -> восстановление пропусков», общий для batch-инференса и веб-сервиса.

Один и тот же код используется:
* в ``scripts/train.py`` — обучение на маскированном train (и на маскированном test);
* в ``scripts/predict.py`` — построение ``submission.csv``;
* в веб-сервисе — восстановление пропусков в ряде, собранном по произвольному полигону.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ndvi.climatology import Climatology
from ndvi.features import build_features
from ndvi.models.gbm import GapModel
from ndvi.paths import ARTIFACTS_DIR
from ndvi.sensors import DEFAULT_OFFSETS, estimate_offsets
from ndvi.validation import make_masked

MODEL_PATH = ARTIFACTS_DIR / "gap_model.pkl"


@dataclass
class Artifacts:
    """Всё, что нужно для инференса: модель, смещения сенсоров, параметры сборки."""

    model: GapModel
    offsets: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OFFSETS))
    meta: dict = field(default_factory=dict)

    def save(self, path=MODEL_PATH) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path=MODEL_PATH) -> "Artifacts":
        return pickle.loads(path.read_bytes())


def build_training_table(df: pd.DataFrame, seeds=(42, 7, 123), frac: float = 0.15,
                         cold_start_frac: float = 0.0) -> pd.DataFrame:
    """Собирает обучающие примеры, многократно скрывая случайные 15 % наблюдений.

    Несколько сидов дают больше примеров при том же объёме данных: каждая точка ряда
    рано или поздно оказывается «скрытой», а признаки для неё считаются по остальным.
    """
    parts = []
    for seed in seeds:
        split = make_masked(df, frac=frac, seed=seed)
        ctx = split.context
        if cold_start_frac > 0:
            rng = np.random.default_rng(seed + 1)
            pids = np.array(sorted(ctx.anon_polygon_id.unique()))
            cold = set(rng.choice(pids, size=max(1, int(cold_start_frac * pids.size)), replace=False))
            gap_years = split.targets.groupby("anon_polygon_id").year.agg(set)
            drop = np.zeros(len(ctx), dtype=bool)
            for pid in cold:
                yrs = gap_years.get(pid, set())
                drop |= (ctx.anon_polygon_id == pid).to_numpy() & (~ctx.year.isin(yrs)).to_numpy()
            ctx = ctx[~drop].copy()
        observed = ctx[ctx.primary_ndvi.notna()]
        offsets = estimate_offsets(observed)
        clim = Climatology().fit(observed)
        feats = build_features(ctx, split.targets, offsets, clim)
        part = feats.merge(split.truth, on=["anon_polygon_id", "date"], how="left")
        part["y_true"] = part.primary_ndvi
        part["seed"] = seed
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit(train_table: pd.DataFrame, params: dict | None = None,
        offsets: dict[str, float] | None = None, meta: dict | None = None) -> Artifacts:
    """Обучает модель на готовой таблице признаков."""
    model = GapModel(params)
    model.fit(train_table, train_table.y_true.to_numpy())
    return Artifacts(model=model, offsets=offsets or dict(DEFAULT_OFFSETS), meta=meta or {})


def predict_gaps(context: pd.DataFrame, targets: pd.DataFrame, artifacts: Artifacts,
                 clim: Climatology | None = None) -> pd.DataFrame:
    """Восстанавливает ``primary_ndvi`` в строках ``targets`` по видимой части ``context``.

    Возвращает таблицу признаков с колонкой ``primary_ndvi_pred``.
    """
    observed = context[context.primary_ndvi.notna()]
    offsets = estimate_offsets(observed) if len(observed) > 200 else artifacts.offsets
    clim = clim or Climatology().fit(observed)
    feats = build_features(context, targets, offsets, clim)
    feats["primary_ndvi_pred"] = artifacts.model.predict(feats)
    return feats


def restore_series(context: pd.DataFrame, artifacts: Artifacts) -> pd.DataFrame:
    """Заполняет все пропуски ряда: строки без ``primary_ndvi`` считаются гэпами.

    Используется веб-сервисом: на вход приходит ряд по произвольному полигону,
    на выходе — тот же ряд с колонками ``primary_ndvi_filled`` и ``is_filled``.
    """
    ctx = context.copy()
    gap_mask = ctx.primary_ndvi.isna()
    out = ctx.copy()
    out["primary_ndvi_filled"] = ctx.primary_ndvi
    out["is_filled"] = False
    if not gap_mask.any() or ctx.primary_ndvi.notna().sum() < 3:
        return out
    targets = ctx.loc[gap_mask, ["anon_polygon_id", "date", "year", "doy", "crop_type"]]
    feats = predict_gaps(ctx, targets, artifacts)
    out.loc[gap_mask, "primary_ndvi_filled"] = feats["primary_ndvi_pred"].to_numpy()
    out.loc[gap_mask, "is_filled"] = True
    return out
