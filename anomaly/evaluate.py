"""Прокси-метрики детекции аномалий (экспертной разметки нет, поэтому оцениваем косвенно).

1. Селективность: доля сезонов с эпизодом, эпизодов на сезон, доля коротких эпизодов.
2. Согласие с организаторами: наблюдения train со статусом «угнетение/критическая» (Z_org < −1) — какая доля
   попала внутрь наших эпизодов (recall), и какая доля наблюдений внутри эпизодов имеет Z_org < −1 (precision).
3. Независимость от сенсора: корреляция доли угнетённых дней сезона с долей наблюдений S2 в сезоне
   (у сырого Z организаторов S2 читает ниже всех — там связь есть по построению).
4. Ранжирование лет: засушливые 2020 и 2024 должны быть в топе по доле сезонов с эпизодом.

Запуск: uv run python -m anomaly.evaluate reports/anomalies [artifacts/anomaly_loose ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DROUGHT_YEARS = (2020, 2024)


def load_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ep = pd.read_csv(path / "episodes.csv")
    se = pd.read_csv(path / "seasons.csv")
    org = pd.read_parquet(path / "organizer_norm.parquet")
    org["date"] = pd.to_datetime(org["date"])
    return ep, se, org


def inside_episode(org: pd.DataFrame, ep: pd.DataFrame) -> np.ndarray:
    """Флаг: наблюдение попадает внутрь какого-либо эпизода своего полигона."""
    flag = np.zeros(len(org), dtype=bool)
    if ep.empty:
        return flag
    by_pid = {p: g for p, g in ep.groupby("pid")}
    for pid, idx in org.groupby("pid").indices.items():
        g = by_pid.get(pid)
        if g is None:
            continue
        d = org["date"].to_numpy()[idx]
        for s, e in zip(pd.to_datetime(g["start"]).to_numpy(), pd.to_datetime(g["end"]).to_numpy()):
            flag[idx] |= (d >= s) & (d <= e)
    return flag


def selectivity(ep: pd.DataFrame, se: pd.DataFrame) -> dict:
    with_ep = ep.groupby(["pid", "year"]).size().shape[0] if len(ep) else 0
    return {"seasons": int(len(se)), "episodes": int(len(ep)), "seasons_with_episode": with_ep / max(len(se), 1),
            "episodes_per_season": len(ep) / max(len(se), 1),
            "share_short_lt14d": float((ep["days"] < 14).mean()) if len(ep) else np.nan,
            "share_few_obs_le2": float((ep["n_obs"] <= 2).mean()) if len(ep) else np.nan,
            "median_days": float(ep["days"].median()) if len(ep) else np.nan}


def agreement(org: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """Recall/precision наших эпизодов относительно точечного статуса организаторов (train-полигоны, все годы)."""
    o = org.dropna(subset=["org_z"]).reset_index(drop=True)
    inside = inside_episode(o, ep)
    depressed = (o["org_z"] < -1).to_numpy()
    critical = (o["org_z"] < -2).to_numpy()
    return {"recall_depressed_points": float(inside[depressed].mean()), "recall_critical_points": float(inside[critical].mean()),
            "precision_points_in_episodes": float(depressed[inside].mean()) if inside.any() else np.nan,
            "base_rate_depressed": float(depressed.mean()), "share_points_inside": float(inside.mean())}


def sensor_independence(org: pd.DataFrame, se: pd.DataFrame) -> dict:
    """Связь доли угнетённых дней сезона с долей S2-наблюдений: у нас и у сырого Z организаторов."""
    s2_share = org.groupby(["pid", "year"])["sensor"].apply(lambda s: float((s == 0).mean())).rename("s2_share")
    org_dep = org.groupby(["pid", "year"])["org_z"].apply(lambda z: float((z < -1).mean())).rename("org_dep")
    m = se.set_index(["pid", "year"]).join(s2_share).join(org_dep).dropna(subset=["s2_share"])
    m = m[m["s2_share"].between(0.05, 0.95)]
    return {"corr_ours_vs_s2share": float(m["share_days_depressed"].corr(m["s2_share"])),
            "corr_org_vs_s2share": float(m["org_dep"].corr(m["s2_share"])), "n_seasons": int(len(m))}


def year_ranking(ep: pd.DataFrame, se: pd.DataFrame) -> dict:
    rate = (ep.groupby("year").apply(lambda g: g[["pid"]].drop_duplicates().shape[0]) / se.groupby("year").size()).fillna(0)
    ranks = rate.rank(ascending=False)
    return {"rate_by_year": {int(y): round(float(v), 3) for y, v in rate.sort_index().items()},
            "drought_years_rank": {int(y): int(ranks.get(y, np.nan)) for y in DROUGHT_YEARS},
            "top3": [int(y) for y in rate.sort_values(ascending=False).index[:3]]}


def evaluate(path: Path) -> dict:
    ep, se, org = load_run(path)
    return {"run": str(path), **selectivity(ep, se), **agreement(org, ep), **sensor_independence(org, se), **year_ranking(ep, se),
            "by_cause": ep["cause"].value_counts().to_dict() if len(ep) else {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Прокси-метрики детекции аномалий")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--save", action="store_true", help="сохранить метрики в <run>/quality.json (их читает сервис)")
    args = parser.parse_args()
    rows = [evaluate(Path(r)) for r in args.runs]
    if args.save:
        for run, row in zip(args.runs, rows):
            (Path(run) / "quality.json").write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
