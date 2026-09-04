"""Детекция и интерпретация негативных аномалий в динамике NDVI.

Три вещи, без которых детекция превращается в шум (все три следуют из EDA):

1. **Гармонизация сенсоров.** Средний z-score в исходных данных: S2 −0.23, Landsat +0.06,
   MODIS +0.52. То есть «аномалия» из коробки во многом отражает, каким спутником снимали,
   а не состояние поля. Поэтому сначала приводим всё к шкале Sentinel-2 и заново считаем
   норму по гармонизированному ряду.
2. **Фильтр одиночных точек.** Из 4 859 наблюдений с z < −1 в train треть изолированы:
   соседи нормальные. Это облака и тени. Эпизодом считаем только устойчивое отклонение.
3. **Отделение уборки от угнетения.** Падение NDVI после пика сезона у озимой пшеницы —
   это жатва, а не стресс. Такие эпизоды помечаются отдельно.

Интерпретация строится по правилам и всегда сопровождается оценкой уверенности: эксперты
оценивают не только факт детекции, но и объяснение.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ndvi.climatology import Climatology
from ndvi.data import NDVI_MAX, NDVI_MIN
from ndvi.sensors import DEFAULT_OFFSETS
from ndvi.smoothing import robust_local_linear

#: пороги по z-score, совпадают с логикой поля ``status`` в исходных данных
Z_DEPRESSED = -1.0
Z_CRITICAL = -2.0
#: минимальная сигма нормы: на плато сезона она бывает 0.01 и z взрывается
MIN_SIGMA = 0.05
#: эпизодом считается серия минимум из стольких наблюдений
MIN_POINTS = 3
#: соседние серии, разделённые не более чем этим числом дней, склеиваются в один эпизод
MERGE_GAP_DAYS = 10

MONTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня",
             "июля", "августа", "сентября", "октября", "ноября", "декабря"]
SENSOR_RU = {"s2": "Sentinel-2", "landsat": "Landsat", "modis": "MODIS", "none": "неизвестный"}


def _fmt_date(d: pd.Timestamp) -> str:
    return f"{d.day} {MONTHS_RU[d.month - 1]}"


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    """Русское согласование числительного: 1 день, 2 дня, 5 дней."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return forms[1]
    return forms[2]


@dataclass
class Episode:
    """Один аномальный эпизод на одном полигоне."""

    anon_polygon_id: str
    year: int
    start: str
    end: str
    duration_days: int
    n_points: int
    min_z: float
    mean_z: float
    ndvi_deficit: float          # насколько NDVI ниже нормы в среднем, в единицах NDVI
    severity: str                # "угнетение" | "критическая"
    kind: str                    # "стресс" | "уборка" | "низкое качество данных"
    confidence: float            # 0..1
    drivers: list[str] = field(default_factory=list)
    explanation: str = ""
    sensors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Подготовка ряда
# --------------------------------------------------------------------------- #

def harmonize(series: pd.DataFrame, offsets: dict[str, float] | None = None,
              value_col: str = "primary_ndvi", src_col: str = "src") -> pd.DataFrame:
    """Приводит NDVI всех сенсоров к шкале Sentinel-2 и клиппит физически невозможные значения."""
    offsets = offsets or DEFAULT_OFFSETS
    out = series.copy()
    off = out[src_col].map(offsets).fillna(0.0) if src_col in out else 0.0
    out["ndvi_s2"] = out[value_col].clip(NDVI_MIN, NDVI_MAX) - off
    return out


def add_zscore(series: pd.DataFrame, clim: Climatology | None = None,
               exclude_current_year: bool = True, smooth_bw: float = 20.0) -> pd.DataFrame:
    """Считает норму по гармонизированному ряду и z-score — сырой и по сглаженной кривой.

    Сглаженный z нужен, чтобы отличить реальное угнетение (просела вся кривая) от
    одиночного облачного провала (просела одна точка, кривая на месте).
    """
    s = series[series.ndvi_s2.notna()].copy()
    if clim is None:
        clim = Climatology().fit(s.assign(primary_ndvi=s.ndvi_s2))

    out = series.copy()
    stats = clim.lookup_frame(out.assign(crop_type=out.get("crop_type", "")),
                              exclude_current_year=exclude_current_year)
    out["norm_mean"] = stats.clim_mean.values
    out["norm_std"] = np.maximum(stats.clim_std.values, MIN_SIGMA)
    out["norm_years"] = stats.clim_n.values
    out["norm_level"] = stats.clim_level.values
    out["z"] = (out.ndvi_s2 - out.norm_mean) / out.norm_std

    # робастная кривая сезона: устойчива к облачным провалам
    out["ndvi_smooth"] = np.nan
    days = (out.date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    for (_, _), idx in out.groupby([out.anon_polygon_id, out.year]).groups.items():
        sub = out.loc[idx]
        m = sub.ndvi_s2.notna().to_numpy()
        if m.sum() >= 4:
            x = days[out.index.get_indexer(idx)][m]
            y = sub.ndvi_s2.to_numpy()[m]
            xt = days[out.index.get_indexer(idx)]
            out.loc[idx, "ndvi_smooth"] = robust_local_linear(x, y, xt, smooth_bw)
    out["z_smooth"] = (out.ndvi_smooth - out.norm_mean) / out.norm_std
    return out


# --------------------------------------------------------------------------- #
# Сборка эпизодов
# --------------------------------------------------------------------------- #

def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Индексы непрерывных серий True: список пар (начало, конец включительно)."""
    out = []
    i = 0
    n = flags.size
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _precip_norm(pol_ctx: pd.DataFrame, doy_from: int, doy_to: int, year: int):
    """Медианная сумма осадков за то же окно doy в другие годы того же полигона."""
    d = pol_ctx[pol_ctx.era5_precip_mm.notna()]
    if d.empty:
        return np.nan
    win = d[(d.doy >= doy_from) & (d.doy <= doy_to)]
    per_year = win[win.year != year].groupby("year").era5_precip_mm.sum()
    return float(per_year.median()) if len(per_year) else np.nan


def _hot_norm(pol_ctx: pd.DataFrame, doy_from: int, doy_to: int, year: int):
    """Медианное число дней жарче 30 °C за то же окно doy в другие годы."""
    d = pol_ctx[pol_ctx.era5_temp_c.notna()]
    if d.empty:
        return np.nan
    win = d[(d.doy >= doy_from) & (d.doy <= doy_to) & (d.year != year)]
    if win.empty:
        return np.nan
    per_year = win.groupby("year").era5_temp_c.apply(lambda s: int((s > 30).sum()))
    return float(per_year.median())


def _peak_doy(clim: Climatology, pid: str, crop: str) -> float:
    """День года, на который приходится максимум климатической нормы полигона."""
    doys = np.arange(95, 300, 5)
    vals = [clim.lookup(pid, crop, int(d))[0] for d in doys]
    vals = np.array(vals, float)
    if not np.isfinite(vals).any():
        return np.nan
    return float(doys[int(np.nanargmax(vals))])


def detect_episodes(series: pd.DataFrame, context: pd.DataFrame | None = None,
                    clim: Climatology | None = None,
                    offsets: dict[str, float] | None = None,
                    min_points: int = MIN_POINTS,
                    z_threshold: float = Z_DEPRESSED) -> tuple[pd.DataFrame, list[Episode]]:
    """Основная функция: ряд -> (обогащённый ряд, список эпизодов).

    ``series`` — наблюдения одного или нескольких полигонов с колонками
    ``anon_polygon_id, date, year, doy, primary_ndvi, src, crop_type`` и, желательно, ERA5.
    ``context`` — данные для климатических норм погоды (по умолчанию сам ``series``).
    """
    context = context if context is not None else series
    s = harmonize(series, offsets)
    if clim is None:
        obs = s[s.ndvi_s2.notna()]
        clim = Climatology().fit(obs.assign(primary_ndvi=obs.ndvi_s2))
    enriched = add_zscore(s, clim)

    # погодные нормы считаются по полигону: индексируем контекст заранее, иначе фильтр
    # по всей таблице на каждый эпизод превращается в узкое место
    ctx_by_pid = {k: v for k, v in context.groupby("anon_polygon_id")}
    empty_ctx = context.iloc[:0]

    episodes: list[Episode] = []
    for (pid, year), d in enriched.groupby(["anon_polygon_id", "year"], sort=True):
        d = d.sort_values("date")
        obs = d[d.ndvi_s2.notna()]
        if len(obs) < min_points + 1:
            continue
        crop = str(obs.crop_type.iloc[0]) if "crop_type" in obs else ""
        peak = _peak_doy(clim, pid, crop)

        # точка считается подозрительной, если ниже порога и по сырому, и по сглаженному z:
        # первый ловит факт, второй отсеивает одиночное облако
        raw_bad = (obs.z < z_threshold).to_numpy()
        smooth_bad = (obs.z_smooth < z_threshold * 0.6).to_numpy()
        flags = raw_bad & np.where(np.isfinite(obs.z_smooth.to_numpy()), smooth_bad, True)

        runs = [r for r in _runs(flags) if (r[1] - r[0] + 1) >= min_points]
        # склеиваем серии, разделённые коротким «нормальным» промежутком
        merged: list[tuple[int, int]] = []
        for r in runs:
            if merged:
                gap_days = (obs.date.iloc[r[0]] - obs.date.iloc[merged[-1][1]]).days
                if gap_days <= MERGE_GAP_DAYS:
                    merged[-1] = (merged[-1][0], r[1])
                    continue
            merged.append(r)

        for a, b in merged:
            seg = obs.iloc[a:b + 1]
            episodes.append(_describe(seg, pid, int(year), crop, peak,
                                      ctx_by_pid.get(pid, empty_ctx), clim))

    return enriched, episodes


def _describe(seg: pd.DataFrame, pid: str, year: int, crop: str, peak_doy: float,
              pol_ctx: pd.DataFrame, clim: Climatology) -> Episode:
    """Собирает описание эпизода и объясняет его правилами."""
    start, end = seg.date.iloc[0], seg.date.iloc[-1]
    duration = int((end - start).days) + 1
    min_z = float(seg.z.min())
    mean_z = float(seg.z.mean())
    deficit = float((seg.norm_mean - seg.ndvi_s2).mean())
    sensors = sorted({str(x) for x in seg.get("src", pd.Series(dtype=str)) if str(x) != "none"})
    severity = "критическая" if min_z <= Z_CRITICAL else "угнетение"

    # --- погодный контекст --------------------------------------------------
    doy_start = int(seg.doy.iloc[0])
    w_from, w_to = max(1, doy_start - 60), doy_start
    cur = pol_ctx[(pol_ctx.year == year) & (pol_ctx.doy >= w_from) & (pol_ctx.doy <= w_to)]
    precip_60 = float(cur.era5_precip_mm.sum()) if cur.era5_precip_mm.notna().any() else np.nan
    precip_norm = _precip_norm(pol_ctx, w_from, w_to, year)
    hot_30 = float((cur[cur.doy >= doy_start - 30].era5_temp_c > 30).sum()) if cur.era5_temp_c.notna().any() else np.nan
    hot_norm = _hot_norm(pol_ctx, max(1, doy_start - 30), doy_start, year)

    drivers, parts = [], []
    parts.append(
        f"{_fmt_date(start)} – {_fmt_date(end)} {year}, {duration} "
        f"{_plural(duration, ('день', 'дня', 'дней'))}, {len(seg)} "
        f"{_plural(len(seg), ('наблюдение', 'наблюдения', 'наблюдений'))}. "
        f"NDVI ниже нормы в среднем на {abs(mean_z):.1f}σ (минимум {min_z:.1f}σ), "
        f"это {deficit:.2f} по шкале NDVI."
    )

    dry = wet = False
    if np.isfinite(precip_60) and np.isfinite(precip_norm) and precip_norm > 1:
        ratio = precip_60 / precip_norm
        parts.append(f"Осадки за 60 дней до эпизода: {precip_60:.0f} мм при норме {precip_norm:.0f} мм "
                     f"({ratio * 100:.0f} % нормы).")
        if ratio < 0.6:
            drivers.append("дефицит осадков")
            dry = True
        elif ratio > 1.6:
            # избыток влаги тоже угнетает посев: вымокание, выпревание, невозможность полевых работ
            drivers.append("избыток осадков")
            wet = True

    hot = False
    if np.isfinite(hot_30) and np.isfinite(hot_norm):
        parts.append(f"Дней жарче 30 °C за предшествующий месяц: {hot_30:.0f} при норме {hot_norm:.0f}.")
        if hot_30 >= hot_norm + 4:
            drivers.append("аномальная жара")
            hot = True

    # --- фаза сезона: падение после пика у зерновых это уборка --------------
    harvest = False
    # жатва зерновых на юге России — конец июня — август; октябрьский спад это уже новый сев
    if (np.isfinite(peak_doy) and doy_start > peak_doy + 20 and 150 <= doy_start <= 240
            and crop in ("озимая пшеница", "зерновые")):
        harvest = True

    # --- качество данных ----------------------------------------------------
    poor_data = len(sensors) <= 1 and len(seg) <= 3
    norm_years = int(seg.norm_years.min()) if "norm_years" in seg else 0
    if norm_years < 3:
        parts.append(f"Норма построена всего по {norm_years} "
                     f"{_plural(norm_years, ('году', 'годам', 'годам'))} — оценка предварительная.")

    if len(sensors) >= 2:
        parts.append("Подтверждено несколькими сенсорами: " + ", ".join(SENSOR_RU.get(x, x) for x in sensors) + ".")
    elif sensors:
        parts.append(f"Все наблюдения эпизода с одного сенсора ({SENSOR_RU.get(sensors[0], sensors[0])}).")

    # осенний спад у озимых — это не угнетение взрослого посева, а всходы нового сева
    autumn_sowing = doy_start >= 250 and crop in ("озимая пшеница", "зерновые")
    if autumn_sowing:
        parts.append("Эпизод приходится на осень: у озимых это период сева и всходов, "
                     "низкий NDVI здесь говорит о плохом развитии всходов, а не о гибели посева.")

    if harvest:
        kind = "уборка"
        parts.append("Эпизод начинается заметно позже пика сезона у зерновой культуры — "
                     "скорее всего это уборка, а не угнетение посева.")
    elif poor_data and not (dry or hot or wet):
        kind = "низкое качество данных"
        parts.append("Мало наблюдений и один источник — возможен остаточный облачный эффект.")
    else:
        kind = "стресс"
        if dry and hot:
            parts.append("Вероятная причина: почвенная засуха на фоне аномальной жары.")
        elif dry:
            parts.append("Вероятная причина: дефицит влаги.")
        elif hot:
            parts.append("Вероятная причина: тепловой стресс.")
        elif wet:
            parts.append("Вероятная причина: переувлажнение — вымокание посева "
                         "или сорванные полевые работы.")
        else:
            parts.append("Погодного объяснения не нашлось: возможны агротехнические причины "
                         "(поздний сев, болезни, смена культуры) или локальное событие.")

    # --- уверенность --------------------------------------------------------
    conf = 0.35
    conf += min(len(seg), 8) / 8 * 0.25          # сколько наблюдений подтверждают
    conf += min(duration, 30) / 30 * 0.15        # насколько долго держится
    conf += 0.15 if len(sensors) >= 2 else 0.0   # подтверждение разными спутниками
    conf += 0.10 if (dry or hot or wet) else 0.0  # согласуется с погодой
    if norm_years < 3:
        conf -= 0.15
    if kind == "низкое качество данных":
        conf -= 0.20
    conf = float(np.clip(conf, 0.05, 0.99))

    return Episode(
        anon_polygon_id=pid, year=year,
        start=str(start.date()), end=str(end.date()),
        duration_days=duration, n_points=int(len(seg)),
        min_z=round(min_z, 2), mean_z=round(mean_z, 2), ndvi_deficit=round(deficit, 3),
        severity=severity, kind=kind, confidence=round(conf, 2),
        drivers=drivers, explanation=" ".join(parts), sensors=sensors,
    )


def episodes_to_frame(episodes: list[Episode]) -> pd.DataFrame:
    """Список эпизодов в таблицу (пустая таблица с нужными колонками, если эпизодов нет)."""
    if not episodes:
        return pd.DataFrame(columns=[f.name for f in Episode.__dataclass_fields__.values()])
    return pd.DataFrame([e.to_dict() for e in episodes])
