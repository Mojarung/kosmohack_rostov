"""Проверка детектора аномалий на train: что он отсеивает и находит ли известные засухи."""

import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ndvi.anomalies import Z_DEPRESSED, detect_episodes, episodes_to_frame, harmonize
from ndvi.data import load_train
from ndvi.paths import REPORTS_DIR

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

t0 = time.time()
tr = load_train()
enriched, episodes = detect_episodes(tr)
eps = episodes_to_frame(episodes)
print(f"обработано {len(tr)} строк, эпизодов найдено: {len(eps)} за {time.time() - t0:.1f} с\n")

obs = enriched[enriched.ndvi_s2.notna()]

print("### 1. Гармонизация сенсоров убирает сенсорный перекос z-score")
before = tr[tr.ndvi_zscore.notna()].groupby("src").ndvi_zscore.mean().round(3)
after = obs.groupby("src").z.mean().round(3)
print(pd.DataFrame({"z в данных организаторов": before, "z после гармонизации": after}).to_string())

print("\n### 2. Фильтр одиночных выбросов")
flagged = obs.z < Z_DEPRESSED
in_episode = pd.Series(False, index=obs.index)
for _, e in eps.iterrows():
    m = ((obs.anon_polygon_id == e.anon_polygon_id)
         & (obs.date >= pd.Timestamp(e.start)) & (obs.date <= pd.Timestamp(e.end)))
    in_episode |= m
print(f"наблюдений с z < -1: {int(flagged.sum())}")
print(f"из них попали в устойчивые эпизоды: {int((flagged & in_episode).sum())} "
      f"({(flagged & in_episode).mean() / max(flagged.mean(), 1e-9) * 100:.0f} %)")
print(f"отсеяно как одиночные провалы (облако/тень): {int((flagged & ~in_episode).sum())}")

print("\n### 3. Эпизоды по годам")
seasons = obs.groupby("year").anon_polygon_id.nunique()
by_year = eps.groupby("year").agg(
    эпизодов=("anon_polygon_id", "size"),
    полей=("anon_polygon_id", "nunique"),
    стресс=("kind", lambda s: int((s == "стресс").sum())),
    мин_z=("min_z", "min"),
)
by_year["доля полей"] = (by_year["полей"] / seasons).round(2)
weather = tr.groupby("year").agg(осадки=("era5_precip_mm", "sum"), темп=("era5_temp_c", "mean"))
weather["осадки"] = (weather["осадки"] / tr.groupby("year").anon_polygon_id.nunique()).round(0)
print(by_year.join(weather.round(1)).to_string())

print("\n### 4. Типы эпизодов")
print(eps.kind.value_counts().to_string())
print("\nдрайверы:")
print(pd.Series([d for lst in eps.drivers for d in lst]).value_counts().to_string())
print(f"\nсредняя уверенность: {eps.confidence.mean():.2f}, "
      f"эпизодов с уверенностью >= 0.8: {int((eps.confidence >= 0.8).sum())}")

print("\n### 5. Демо-кейсы: засушливые 2020 и 2024")
for y in (2020, 2024):
    sub = eps[(eps.year == y) & (eps.kind == "стресс")].sort_values("confidence", ascending=False)
    print(f"\n{y}: {len(sub)} стрессовых эпизодов на {sub.anon_polygon_id.nunique()} полях")
    if len(sub):
        print("  " + sub.iloc[0].explanation)

eps.to_csv(REPORTS_DIR / "anomaly_episodes_train.csv", index=False, encoding="utf-8")
print(f"\nэпизоды сохранены: reports/anomaly_episodes_train.csv, всего {time.time() - t0:.1f} с")
