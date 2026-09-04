"""Определение примерного положения полигонов датасета по «отпечатку» погоды.

В данных соревнования координат нет — только анонимные ``AOI-xxxx``. Зато есть суточные
ряды ERA5 (температура и осадки), а погода — это география: суммы осадков по дням за
сезон практически уникальны для места.

Метод: берём регулярную сетку точек, качаем для каждой архив ERA5 (Open-Meteo) за тот же
период и ищем точку с максимальным соответствием ряду полигона.

**Важно о точности.** Ряды в датасете — не точечная выборка из ERA5-Land: лучшее совпадение
даёт корреляцию осадков 0.72–0.87 и RMSE температуры около 0.6 °C, а поверхность соответствия
пологая. Скорее всего, организаторы усредняли погоду по полигону или брали другой продукт
реанализа.

Точность измерена прямо: положение каждого полигона оценивалось дважды по двум непересекающимся
половинам сезона 2025 (апрель–середина июля и середина июля–октябрь). Расхождение двух оценок:
медиана 61 км, 75-й процентиль 97 км; для случайных пар точек в этой же области — 257 км.
Значит, сигнал настоящий, но собственная ошибка одной оценки порядка **50–60 км**.

Практический вывод: рисовать по этим числам контур поля нельзя, и отдельное поле на карте
показывать бессмысленно. Осмысленно показывать **скопления**: в каких районах области
сосредоточены обучающие поля и где лежат те, что нужно предсказать.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from ndvi.paths import ARTIFACTS_DIR, CACHE_DIR

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GRID_CACHE = CACHE_DIR / "era5_grid"

#: рабочая область поиска: Ростовская область с запасом на соседние регионы
DEFAULT_BBOX = (37.8, 45.6, 44.6, 50.0)   # minlon, minlat, maxlon, maxlat
#: шаг сетки: мельче нет смысла, поверхность соответствия пологая
DEFAULT_STEP = 0.4
#: собственная ошибка одной оценки, км (измерена по двум половинам сезона)
UNCERTAINTY_KM = 60.0
#: окно сопоставления — вегетационный сезон 2025, он есть у всех тестовых полигонов
DEFAULT_WINDOW = ("2025-04-01", "2025-10-31")


def build_grid(bbox=DEFAULT_BBOX, step: float = DEFAULT_STEP) -> list[tuple[float, float]]:
    """Узлы регулярной сетки внутри bbox."""
    minlon, minlat, maxlon, maxlat = bbox
    lats = np.round(np.arange(minlat, maxlat + 1e-9, step), 2)
    lons = np.round(np.arange(minlon, maxlon + 1e-9, step), 2)
    return [(float(a), float(b)) for a in lats for b in lons]


def _cache_path(lat: float, lon: float, start: str, end: str):
    GRID_CACHE.mkdir(parents=True, exist_ok=True)
    return GRID_CACHE / f"{lat:.2f}_{lon:.2f}_{start}_{end}.json"


def fetch_grid(points, start: str, end: str, chunk: int = 8, pause: float = 2.0,
               max_retry: int = 5, verbose: bool = True) -> dict:
    """Качает погоду для узлов сетки. Каждый узел кэшируется отдельным файлом.

    Open-Meteo ограничивает частоту запросов, поэтому узлы идут пачками с паузой и
    экспоненциальным откатом при 429.
    """
    import httpx

    out, need = {}, []
    for la, lo in points:
        p = _cache_path(la, lo, start, end)
        if p.exists():
            try:
                d = json.loads(p.read_text())
                out[(la, lo)] = pd.DataFrame({"t": d["t"], "p": d["p"]}, index=d["time"])
                continue
            except Exception:
                pass
        need.append((la, lo))
    if verbose:
        print(f"сетка: из кэша {len(out)}, качаем {len(need)}", flush=True)

    with httpx.Client(timeout=180) as cl:
        for i in range(0, len(need), chunk):
            part = need[i:i + chunk]
            delay = pause
            for attempt in range(max_retry):
                try:
                    r = cl.get(ARCHIVE_URL, params={
                        "latitude": ",".join(f"{a:.2f}" for a, _ in part),
                        "longitude": ",".join(f"{b:.2f}" for _, b in part),
                        "start_date": start, "end_date": end,
                        "daily": "temperature_2m_mean,precipitation_sum", "timezone": "UTC"})
                    r.raise_for_status()
                    js = r.json()
                    if isinstance(js, dict):
                        js = [js]
                    for (la, lo), rec in zip(part, js):
                        dd = rec["daily"]
                        _cache_path(la, lo, start, end).write_text(json.dumps(
                            {"time": dd["time"], "t": dd["temperature_2m_mean"],
                             "p": dd["precipitation_sum"]}))
                        out[(la, lo)] = pd.DataFrame(
                            {"t": dd["temperature_2m_mean"], "p": dd["precipitation_sum"]},
                            index=dd["time"])
                    break
                except Exception as exc:
                    if attempt == max_retry - 1:
                        log.warning("узлы %s пропущены: %s", part, exc)
                        break
                    time.sleep(delay)
                    delay *= 2
            time.sleep(pause)
            if verbose and (i // chunk) % 5 == 0:
                print(f"  {min(i + chunk, len(need))}/{len(need)}", flush=True)
    return out


def _series(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame | None:
    """Ряд погоды полигона в виде (индекс — дата, колонки t и p)."""
    d = df.dropna(subset=["era5_temp_c"])
    d = d[(d.date >= start) & (d.date <= end)]
    if len(d) < 60:
        return None
    return pd.DataFrame({"t": d.era5_temp_c.values, "p": d.era5_precip_mm.values},
                        index=d.date.dt.strftime("%Y-%m-%d").values)


def match_series(target: pd.DataFrame, grid: dict, w_precip: float = 2.0) -> pd.DataFrame:
    """Сопоставляет ряд полигона со всеми узлами сетки.

    Температура почти одинакова везде, различает именно осадки, поэтому их корреляция
    входит в счёт с большим весом.
    """
    rows = []
    for (la, lo), g in grid.items():
        m = g.reindex(target.index)
        ok = m.t.notna() & target.t.notna()
        if ok.sum() < 60:
            continue
        rt = float(np.corrcoef(m.t[ok], target.t[ok])[0, 1])
        et = float(np.sqrt(((m.t[ok] - target.t[ok]) ** 2).mean()))
        okp = m.p.notna() & target.p.notna()
        rp = float(np.corrcoef(m.p[okp], target.p[okp])[0, 1]) if okp.sum() >= 60 else np.nan
        rows.append((la, lo, rt, et, rp, rt + w_precip * (rp if np.isfinite(rp) else 0.0)))
    return pd.DataFrame(rows, columns=["lat", "lon", "corr_t", "rmse_t", "corr_p", "score"])


def locate_polygons(df: pd.DataFrame, grid: dict, window=DEFAULT_WINDOW,
                    verbose: bool = True) -> pd.DataFrame:
    """Оценивает положение каждого полигона таблицы. Полигоны без ERA5 пропускаются."""
    start, end = window
    rows = []
    for pid, sub in df.groupby("anon_polygon_id"):
        target = _series(sub, start, end)
        if target is None:
            continue
        res = match_series(target, grid)
        if res.empty:
            continue
        best = res.sort_values("score", ascending=False).iloc[0]
        # насколько уверенно выбран узел: отрыв лидера от медианы поля соответствия
        margin = float(best.score - res.score.median())
        rows.append({"anon_polygon_id": pid, "lat": float(best.lat), "lon": float(best.lon),
                     "corr_t": round(float(best.corr_t), 4), "rmse_t": round(float(best.rmse_t), 3),
                     "corr_p": round(float(best.corr_p), 4), "margin": round(margin, 3),
                     "n_days": int(len(target))})
        if verbose:
            print(f"  {pid}: {best.lat:.2f}, {best.lon:.2f}  "
                  f"corr_p={best.corr_p:.3f} rmse_t={best.rmse_t:.2f}", flush=True)
    return pd.DataFrame(rows)


def save(locations: pd.DataFrame, roles: dict[str, str], path=None) -> dict:
    """Сохраняет результат вместе с ролями полигонов (обучение/предсказание)."""
    path = path or ARTIFACTS_DIR / "polygon_locations.json"
    payload = {
        "note": ("Положение оценено по совпадению суточных рядов ERA5 с архивом Open-Meteo. "
                 f"Собственная ошибка оценки порядка {UNCERTAINTY_KM:.0f} км — это район, "
                 "а не контур поля: координат в данных соревнования нет."),
        "uncertainty_km": UNCERTAINTY_KM,
        "window": list(DEFAULT_WINDOW),
        "items": [{**r, "role": roles.get(r["anon_polygon_id"], "predict")}
                  for r in locations.to_dict("records")],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return payload
