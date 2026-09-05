"""Слой данных веб-сервиса: ряды полигонов, кривые, нормы, эпизоды и восстановленные значения для UI.

Полигоны из train/test анонимны (координат нет), поэтому они показываются списком; новые территории
приходят через сбор данных (service.collect) и анализируются тем же кодом anomaly/.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from anomaly.climatology import crop_norms, norm_for_year
from anomaly.config import REPORT_DIR
from anomaly.detect import z_series
from anomaly.series import curves_by_year, harmonized_series
from anomaly.weather import daily_weather
from gapfill.config import CROP_CODES, ROOT, SUBMISSION_PATH
from gapfill.data import load_all, polygon_kinds
from service.weather_metrics import _numbers

CROP_NAMES = {v: k for k, v in CROP_CODES.items()}
SENSOR_NAMES = {0: "Sentinel-2", 1: "Landsat", 2: "MODIS"}
KIND_NAMES = {"old": "полигон train (история 2010–2024, сезон 2025 из первой версии test)",
              "new_hist": "полигон test с историей", "new_2025only": "полигон test, только сезон 2025"}
# Предсказания для контрольных точек первой версии test (их показываем как восстановленные, но не оцениваем)
EXTRA_SUBMISSIONS = sorted((ROOT / "reports" / "gapfill").glob("submission_*.csv"))


class Store:
    """Ленивая загрузка данных и кэш вычислений по полигонам."""

    def __init__(self) -> None:
        self.obs, self.grid, self.gaps = load_all()
        self.kinds = polygon_kinds(self.obs)
        self.crop_of = self.obs.groupby("pid")["crop"].first().to_dict()
        self.episodes = self._read_csv(REPORT_DIR / "episodes.csv")
        self.seasons = self._read_csv(REPORT_DIR / "seasons.csv")
        self.shape = self._read_csv(REPORT_DIR / "seasons_shape.csv")
        self.restored = self._read_submission()
        self._curves_all = None

    @staticmethod
    def _read_csv(path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _read_submission(self) -> pd.DataFrame:
        """Восстановленные значения контрольных точек: submission.csv (текущий test) плюс старые версии;
        оставляем только строки, которые действительно являются контрольными точками в данных."""
        frames = []
        for path in [SUBMISSION_PATH, *EXTRA_SUBMISSIONS]:
            if path.exists():
                sub = pd.read_csv(path).rename(columns={"anon_polygon_id": "pid", "primary_ndvi_pred": "value"})
                frames.append(sub.assign(date=pd.to_datetime(sub["date"])))
        if not frames:
            return pd.DataFrame(columns=["pid", "date", "value"])
        sub = pd.concat(frames, ignore_index=True).drop_duplicates(["pid", "date"], keep="first")
        return sub.merge(self.grid.loc[self.grid["is_gap"], ["pid", "date"]], on=["pid", "date"], how="inner")

    def crop_norm_for(self, pid: str):
        """Норма по культуре (для полигонов без истории), считается один раз по всем полигонам."""
        if self._curves_all is None:
            self._curves_all = {p: curves_by_year(harmonized_series(g)) for p, g in self.obs.groupby("pid")}
            self._cnorms = crop_norms(self._curves_all, self.crop_of)
        return self._cnorms.get(self.crop_of[pid])

    def polygons(self) -> list[dict]:
        """Список полигонов с краткой сводкой для панели выбора."""
        out = []
        ep_by = self.episodes.groupby("pid") if len(self.episodes) else None
        for pid, g in self.obs.groupby("pid"):
            eps = ep_by.get_group(pid) if ep_by is not None and pid in ep_by.groups else pd.DataFrame()
            out.append({"pid": pid, "crop": CROP_NAMES.get(int(self.crop_of[pid]), "?"),
                        "kind": KIND_NAMES.get(self.kinds[pid], self.kinds[pid]),
                        "years": [int(y) for y in sorted(g["year"].unique())], "n_obs": len(g),
                        "n_episodes": len(eps),
                        "n_critical": int((eps["severity"] == "критическая").sum()) if len(eps) else 0,
                        "has_weather": bool(self.grid.loc[self.grid["pid"] == pid, "era5_temp_c"].notna().any()),
                        "n_gaps": int((self.restored["pid"] == pid).sum())})
        return out

    @lru_cache(maxsize=128)  # noqa: B019 — единственный Store приложения
    def polygon(self, pid: str) -> dict:
        """Полный набор для графиков полигона: по годам наблюдения, кривая, норма, Z; эпизоды; погода."""
        rows = self.obs.loc[self.obs["pid"] == pid]
        series = harmonized_series(rows)
        curves = curves_by_year(series)
        cnorm = self.crop_norm_for(pid) if len(curves) < 4 else None
        restored = self.restored.loc[self.restored["pid"] == pid]
        years = {}
        for year, curve in curves.items():
            norm, source = norm_for_year(curves, year, cnorm)
            zs = z_series(curve, norm)
            obs_year = series.loc[series["year"] == year]
            years[int(year)] = {
                "norm_source": source,
                "observations": [{"date": d.strftime("%Y-%m-%d"), "value": float(v), "harmonized": float(h),
                                  "sensor": SENSOR_NAMES[int(s)], "artifact": bool(a)}
                                 for d, v, h, s, a in zip(obs_year["date"], obs_year["primary_ndvi"], obs_year["h"],
                                                          obs_year["sensor"], obs_year["artifact"])],
                "restored": [{"date": d.strftime("%Y-%m-%d"), "value": float(v)}
                             for d, v in zip(restored["date"], restored["value"]) if d.year == year],
                "curve": _series_json(zs, "value"), "norm_mean": _series_json(zs, "norm_mean"),
                "norm_std": _series_json(zs, "norm_std"), "z": _series_json(zs, "z"),
            }
        eps = self.episodes.loc[self.episodes["pid"] == pid] if len(self.episodes) else pd.DataFrame()
        shape = self.shape.loc[self.shape["pid"] == pid] if len(self.shape) else pd.DataFrame()
        return {"pid": pid, "crop": CROP_NAMES.get(int(self.crop_of[pid]), "?"), "kind": KIND_NAMES.get(self.kinds[pid]),
                "years": years, "episodes": _records(eps), "shape": _records(shape[shape["shape_flag"]]) if len(shape) else [],
                "weather": self.weather(pid)}

    def weather(self, pid: str) -> dict:
        """Ежедневная погода полигона по годам (пусто, если ERA5 нет)."""
        w = daily_weather(self.grid.loc[self.grid["pid"] == pid])
        if w is None:
            return {}
        out = {}
        for year, g in w.groupby("year"):
            g = g.dropna(subset=["era5_temp_c"])
            out[int(year)] = {"date": [d.strftime("%Y-%m-%d") for d in g["date"]],
                              "temp": _numbers(g["era5_temp_c"]),
                              "precip": _numbers(g["era5_precip_mm"])}
        return out


def _series_json(zs: pd.DataFrame, col: str) -> list:
    ok = zs["weight"] >= 0.8 if col in ("value", "z") else zs[col].notna()
    return [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
            for d, v, k in zip(zs["date"], zs[col], ok) if k and np.isfinite(v)]


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        if isinstance(clean.get("weather"), str):
            try:
                clean["weather"] = json.loads(clean["weather"])
            except json.JSONDecodeError:
                pass
        out.append(clean)
    return out
