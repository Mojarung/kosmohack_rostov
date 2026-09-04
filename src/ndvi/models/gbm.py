"""Градиентный бустинг поверх признаков гэпа.

Модель учит не само значение NDVI, а **поправку к сенсорно-осведомлённой интерполяции**
(``base_interp``). Для деревьев это принципиально: их выход кусочно-постоянный, поэтому
предсказывать гладко меняющийся уровень NDVI напрямую они умеют плохо, а предсказывать
небольшую поправку к разумной опоре — хорошо.

Целевая функция по умолчанию — квадратичная, потому что метрика соревнования RMSE.
Huber доступен как альтернатива (4 % выбросов дают 41 % MSE, и есть соблазн их не ловить),
сравнение обеих версий лежит в ``reports/experiments.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

#: колонки таблицы признаков, которые не подаются в модель
META_COLS = ("anon_polygon_id", "date", "hidden_src", "crop_type",
             "primary_ndvi", "src_true", "y_true", "seed", "is_cold_start",
             "primary_ndvi_pred", "is_synthetic_gap")

CATEGORICAL = ["crop_type", "hidden_src"]

DEFAULT_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42,
)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Числовые признаки + категориальные, в фиксированном порядке."""
    num = [c for c in feats.columns
           if c not in META_COLS and pd.api.types.is_numeric_dtype(feats[c])]
    return num + [c for c in CATEGORICAL if c in feats.columns]


def make_matrix(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Приводит категориальные колонки к типу ``category`` — sklearn умеет их нативно."""
    x = feats[cols].copy()
    for c in CATEGORICAL:
        if c in x.columns:
            x[c] = x[c].astype("category")
    return x


class GapModel:
    """Обёртка: хранит список признаков, базовую опору и саму модель."""

    def __init__(self, params: dict | None = None, base_col: str = "base_interp"):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.base_col = base_col
        self.cols: list[str] = []
        self.model: HistGradientBoostingRegressor | None = None
        self.categories: dict[str, list] = {}

    def _base(self, feats: pd.DataFrame) -> np.ndarray:
        """Опорный прогноз с откатами, если сглаженная кривая или соседи отсутствуют."""
        base = feats[self.base_col].to_numpy(float).copy()
        for fallback in ("base_interp", "base_mean2", "base_smooth", "clim_mean"):
            if fallback in feats:
                v = feats[fallback].to_numpy(float)
                bad = ~np.isfinite(base)
                base[bad] = v[bad]
        base[~np.isfinite(base)] = 0.35  # последний откат: примерно медиана NDVI по данным
        return base

    def fit(self, feats: pd.DataFrame, y: np.ndarray) -> "GapModel":
        self.cols = feature_columns(feats)
        x = make_matrix(feats, self.cols)
        self.categories = {c: list(x[c].cat.categories) for c in CATEGORICAL if c in x.columns}
        resid = np.asarray(y, float) - self._base(feats)
        ok = np.isfinite(resid)
        self.model = HistGradientBoostingRegressor(
            categorical_features=[c for c in CATEGORICAL if c in x.columns], **self.params)
        self.model.fit(x[ok], resid[ok])
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        x = make_matrix(feats, self.cols)
        for c, cats in self.categories.items():
            if c in x.columns:
                x[c] = pd.Categorical(x[c], categories=cats)
        pred = self._base(feats) + self.model.predict(x)
        return np.clip(pred, -0.2, 1.0)
