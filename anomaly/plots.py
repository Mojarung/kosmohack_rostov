"""Графики эпизодов для отчёта и презентации: ряд полигона, кривая, норма, Z-score, погода.

Запуск: uv run python -m anomaly.plots AOI-0043:2019 AOI-0065:2024 [--top 6]
Без аргументов рисует --top самых тяжёлых эпизодов из reports/anomalies/episodes.csv.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from anomaly.climatology import crop_norms, norm_for_year
from anomaly.config import FIG_DIR, REPORT_DIR
from anomaly.detect import z_series
from anomaly.series import curves_by_year, harmonized_series
from anomaly.weather import daily_weather
from gapfill.data import load_all

SENSOR_STYLE = {0: ("Sentinel-2", "#2b6cb0"), 1: ("Landsat", "#805ad5"), 2: ("MODIS", "#dd6b20")}


def _season_frames(obs, grid, pid, year):
    """Кривая, норма, Z, наблюдения и погода сезона (норма по культуре, если истории нет)."""
    crop_of = obs.groupby("pid")["crop"].first().to_dict()
    series = harmonized_series(obs.loc[obs["pid"] == pid])
    curves = curves_by_year(series)
    cnorm = None
    if int((year not in curves) or (len(curves) < 4)):
        all_curves = {p: curves_by_year(harmonized_series(g)) for p, g in obs.groupby("pid")}
        cnorm = crop_norms(all_curves, crop_of).get(crop_of[pid])
    norm, source = norm_for_year(curves, year, cnorm)
    zs = z_series(curves[year], norm)
    return series.loc[series["year"] == year], zs, source, daily_weather(grid.loc[grid["pid"] == pid])


def plot_season(obs, grid, episodes: pd.DataFrame, pid: str, year: int) -> str:
    """Три панели: NDVI (наблюдения по сенсорам, кривая, норма ±1σ, эпизоды), Z-score, погода."""
    s, zs, source, wx = _season_frames(obs, grid, pid, year)
    eps = episodes.loc[(episodes["pid"] == pid) & (episodes["year"] == year)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, height_ratios=[3, 1.3, 1.3])
    ax = axes[0]
    ax.fill_between(zs["date"], zs["norm_mean"] - zs["norm_std"], zs["norm_mean"] + zs["norm_std"], color="gray", alpha=0.2,
                    label=f"норма ±1σ ({source})")
    ax.plot(zs["date"], zs["norm_mean"], color="gray", lw=1.2)
    ok = zs["weight"] >= 0.8
    ax.plot(zs["date"].where(ok), zs["value"].where(ok), color="black", lw=1.8, label="восстановленная кривая сезона")
    for code, (name, color) in SENSOR_STYLE.items():
        sel = (s["sensor"] == code) & ~s["artifact"]
        ax.scatter(s.loc[sel, "date"], s.loc[sel, "h"], s=22, color=color, label=f"{name} (в шкале S2)", zorder=3)
    art = s["artifact"]
    if art.any():
        ax.scatter(s.loc[art, "date"], s.loc[art, "h"], s=40, marker="x", color="red", label="артефакт (облако/тень)", zorder=4)
    for _, e in eps.iterrows():
        color = "#c8423f" if e["severity"] == "критическая" else "#e8a33d"
        for a in axes[:2]:
            a.axvspan(pd.Timestamp(e["start"]), pd.Timestamp(e["end"]), color=color, alpha=0.18)
        ax.annotate(e["cause"], (pd.Timestamp(e["start"]), 0.92), fontsize=8, color=color)
    ax.set(ylabel="NDVI", ylim=(0, 1), title=f"{pid}, сезон {year}: эпизоды угнетения и их причины")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].plot(zs["date"], zs["z"], color="black", lw=1.2)
    for lvl, c in ((-1, "#e8a33d"), (-2, "#c8423f")):
        axes[1].axhline(lvl, ls="--", color=c, lw=1)
    axes[1].set(ylabel="Z-score", ylim=(-4, 3))
    if wx is not None:
        w = wx.loc[wx["year"] == year]
        axes[2].bar(w["date"], w["era5_precip_mm"], color="#2b6cb0", width=1, label="осадки, мм/день")
        ax2 = axes[2].twinx()
        ax2.plot(w["date"], w["era5_temp_c"], color="#dd6b20", lw=1, label="температура, °C")
        axes[2].set(ylabel="осадки, мм")
        ax2.set_ylabel("°C")
    else:
        axes[2].text(0.5, 0.5, "ERA5 для полигона нет", ha="center", transform=axes[2].transAxes)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{pid}_{year}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Графики эпизодов")
    parser.add_argument("targets", nargs="*", help="polygon:year")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()
    obs, grid, _ = load_all()
    episodes = pd.read_csv(REPORT_DIR / "episodes.csv")
    targets = [(t.split(":")[0], int(t.split(":")[1])) for t in args.targets]
    if not targets:
        top = episodes.sort_values("min_z").drop_duplicates(["pid", "year"]).head(args.top)
        targets = list(zip(top["pid"], top["year"]))
    for pid, year in targets:
        print(plot_season(obs, grid, episodes, pid, year))


if __name__ == "__main__":
    main()
