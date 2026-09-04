"""Пакетная детекция аномалий по всем полигонам train + test и сверка со статусами организаторов.

Два прохода: сначала для каждого сезона строятся кривая, норма и ежедневный Z; по всем полигонам собирается
региональный контекст (Z других полей в те же даты) и региональная погода (медиана ERA5 полей с погодой).
Затем выделяются эпизоды и объясняются с учётом контекста.

Запуск: uv run python -m anomaly.run [--polygons AOI-0043 AOI-0065] [--loose] [--mask-share 0.15]
Результаты: episodes.csv (эпизоды с причинами и текстом), seasons.csv (фенометрики), summary.json,
organizer_norm.parquet (воспроизведённая норма организаторов).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly.climatology import crop_norms, norm_for_year, organizer_norm, status_from_z
from anomaly.config import REPORT_DIR
from anomaly.detect import find_episodes, norm_phenology, phenology, phenology_deviation, z_series
from anomaly.report import classify, describe, severity_label
from anomaly.series import curves_by_year, harmonized_series
from anomaly.weather import daily_weather, episode_weather, regional_weather
from gapfill.data import load_all, make_mask

MIN_REGION_POLYGONS = 3


@dataclass
class SeasonContext:
    """Всё, что известно о сезоне полигона до выделения эпизодов."""
    pid: str
    year: int
    zs: pd.DataFrame
    norm_source: str
    pheno: dict
    dev: dict
    obs_year: pd.DataFrame


def polygon_seasons(pid: str, series: pd.DataFrame, crop_norm: pd.DataFrame | None) -> list[SeasonContext]:
    """Кривые, нормы, Z и фенометрики всех сезонов полигона."""
    curves = curves_by_year(series)
    out = []
    for year, curve in curves.items():
        norm, source = norm_for_year(curves, year, crop_norm)
        if norm["norm_mean"].notna().sum() < 30:
            continue
        pheno = phenology(curve)
        dev = phenology_deviation(pheno, norm_phenology(norm))
        out.append(SeasonContext(pid, year, z_series(curve, norm), source, pheno, dev,
                                 series.loc[series["year"] == year]))
    return out


def regional_z(seasons: list[SeasonContext]) -> pd.DataFrame:
    """Все валидные ежедневные Z всех полигонов: колонки date, z, pid."""
    parts = [s.zs[["date", "z"]].assign(pid=s.pid) for s in seasons]
    return pd.concat(parts, ignore_index=True).dropna(subset=["z"])


def region_context(all_z: pd.DataFrame, pid: str, start: str, end: str) -> dict:
    """Медианный Z других полигонов в дни эпизода и доля полей ниже нормы: региональное или локальное явление."""
    sel = all_z.loc[(all_z["pid"] != pid) & (all_z["date"] >= pd.Timestamp(start)) & (all_z["date"] <= pd.Timestamp(end))]
    if sel.empty or sel["pid"].nunique() < MIN_REGION_POLYGONS:
        return {"available": False}
    per_day = sel.groupby("date")["z"].median()
    per_poly = sel.groupby("pid")["z"].mean()
    return {"available": True, "region_z": float(per_day.mean()), "n_polygons": int(len(per_poly)),
            "share_polygons_depressed": float((per_poly < -1).mean())}


def season_row(s: SeasonContext) -> dict:
    return {"pid": s.pid, "year": s.year, "norm_source": s.norm_source, "n_obs": int(len(s.obs_year)),
            "n_artifacts": int(s.obs_year["artifact"].sum()), "mean_z": float(np.nanmean(s.zs["z"])),
            "share_days_depressed": float(np.nanmean(s.zs["z"] < -1)),
            **{f"ph_{k}": v for k, v in s.pheno.items()}, **{f"dev_{k}": v for k, v in s.dev.items()}}


def season_episodes(s: SeasonContext, weather: pd.DataFrame | None, weather_source: str, all_z: pd.DataFrame,
                    strict: bool) -> list[dict]:
    """Эпизоды сезона с погодой, региональным контекстом, причиной и текстом."""
    obs_days = s.obs_year.loc[~s.obs_year["artifact"], "day_num"].to_numpy()
    rows = []
    for ep in find_episodes(s.zs, obs_days, strict):
        window = ((s.obs_year["date"] >= pd.Timestamp(ep["start"]) - pd.Timedelta(days=10))
                  & (s.obs_year["date"] <= pd.Timestamp(ep["end"]) + pd.Timedelta(days=10)))
        near = int((s.obs_year["artifact"] & window).sum())
        wx = episode_weather(weather, ep["start"], ep["end"]) | {"source": weather_source}
        region = region_context(all_z, s.pid, ep["start"], ep["end"])
        cause, conf, reasons = classify(ep, s.dev, s.pheno, wx, near, region)
        rows.append({"pid": s.pid, "year": s.year, **ep, "severity": severity_label(ep), "cause": cause,
                     "confidence": conf, "norm_source": s.norm_source, "weather_source": weather_source,
                     "artifacts_near": near, "region_z": region.get("region_z"),
                     "region_share_depressed": region.get("share_polygons_depressed"),
                     "weather": json.dumps(wx, ensure_ascii=False), "reasons": " | ".join(reasons),
                     "text": describe(s.pid, s.year, ep, cause, conf, reasons, s.norm_source)})
    return rows


def compare_with_org(org: pd.DataFrame, train_rows: pd.DataFrame) -> dict:
    """Сверка воспроизведённой нормы и статуса с колонками организаторов в train."""
    m = org.merge(train_rows[["pid", "date", "ndvi_zscore", "status", "ndvi_climatology_mean"]], on=["pid", "date"], how="inner")
    m = m.dropna(subset=["ndvi_zscore", "org_z"])
    our_status = status_from_z(m["org_z"].to_numpy())
    return {"n": int(len(m)), "clim_mean_rmse": float(np.sqrt(np.mean((m["org_mean"] - m["ndvi_climatology_mean"]) ** 2))),
            "z_corr": float(np.corrcoef(m["org_z"], m["ndvi_zscore"])[0, 1]),
            "status_agreement": float((our_status == m["status"].to_numpy()).mean())}


def run_all(obs: pd.DataFrame, grid: pd.DataFrame, pids: list[str], strict: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полный цикл по списку полигонов; контекст (нормы по культуре, региональный Z и погода) — по всем полигонам."""
    crop_of = obs.groupby("pid")["crop"].first().to_dict()
    series_all = {pid: harmonized_series(g) for pid, g in obs.groupby("pid")}
    cnorms = crop_norms({pid: curves_by_year(s) for pid, s in series_all.items()}, crop_of)
    seasons = [s for pid, ser in series_all.items() for s in polygon_seasons(pid, ser, cnorms.get(crop_of[pid]))]
    all_z = regional_z(seasons)
    region_wx = regional_weather(grid)
    episodes, wanted = [], set(pids)
    for pid in pids:
        wx = daily_weather(grid.loc[grid["pid"] == pid])
        wx, source = (wx, "ERA5 полигона") if wx is not None else (region_wx, "ERA5 региона (медиана соседних полей)")
        for s in seasons:
            if s.pid == pid:
                episodes += season_episodes(s, wx, source, all_z, strict)
    org = pd.concat([organizer_norm(series_all[pid]) for pid in pids], ignore_index=True)
    return (pd.DataFrame(episodes), pd.DataFrame([season_row(s) for s in seasons if s.pid in wanted]),
            org[["pid", "date", "year", "sensor", "org_mean", "org_std", "org_z"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Детекция аномалий вегетации")
    parser.add_argument("--polygons", nargs="*", help="ограничить список полигонов (контекст всё равно по всем)")
    parser.add_argument("--loose", action="store_true", help="без критерия устойчивости/силы (для сравнения)")
    parser.add_argument("--out", type=str, default=str(REPORT_DIR), help="папка результатов")
    parser.add_argument("--mask-share", type=float, default=0.0, help="скрыть долю наблюдений (проверка устойчивости)")
    parser.add_argument("--mask-seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out)
    obs, grid, _ = load_all()
    if args.mask_share > 0:
        obs = obs.loc[~make_mask(obs, share=args.mask_share, seed=args.mask_seed)].reset_index(drop=True)
    train_raw = pd.read_csv("data/train_dataset.csv", low_memory=False).rename(columns={"anon_polygon_id": "pid"})
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    pids = args.polygons or sorted(obs["pid"].unique())
    episodes, seasons, org = run_all(obs, grid, pids, strict=not args.loose)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out_dir / "episodes.csv", index=False, encoding="utf-8")
    seasons.to_csv(out_dir / "seasons.csv", index=False, encoding="utf-8")
    org.to_parquet(out_dir / "organizer_norm.parquet")
    summary = {"n_polygons": len(pids), "n_seasons": int(len(seasons)), "n_episodes": int(len(episodes)),
               "episodes_by_cause": episodes["cause"].value_counts().to_dict() if len(episodes) else {},
               "episodes_by_severity": episodes["severity"].value_counts().to_dict() if len(episodes) else {},
               "org_check": compare_with_org(org, train_raw)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
