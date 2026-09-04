"""Работа с тремя сенсорами: смещения между ними и угадывание сенсора скрытой точки.

Из EDA: Landsat читает на +0.037 выше S2, MODIS на +0.083. Целевой ряд ``primary_ndvi``
склеен из трёх сенсоров, поэтому скрытая точка принадлежит конкретному сенсору, и
«среднее соседей» систематически промахивается, когда сенсоры соседей и гэпа разные.

Сенсор скрытой точки угадывается двумя способами.

1. **По расписанию самого полигона** (запасной вариант): MODIS всегда на doy = 1 (mod 16),
   Landsat с шагом 8 дней от других Landsat-дат, Sentinel-2 с шагом 5 дней. Около 81 % попаданий.
2. **По календарю съёмки всей области** (основной): спутник снимает полосу земли целиком,
   поэтому если в тот же день Landsat снял соседнее поле, значит он прошёл и над нашим.
   Поля на одном витке съёмки распознаются по совпадению дат наблюдений. Это даёт **97.6 %**.

Второй способ работает только на данных задачи и ничего не требует извне: календарь
восстанавливается из тех же таблиц, просто по другим полигонам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndvi.data import SENSORS

#: запасные значения смещений (из EDA на train), если оценить не на чем
DEFAULT_OFFSETS = {"s2": 0.0, "landsat": 0.037, "modis": 0.083, "none": 0.0}


def estimate_offsets(obs: pd.DataFrame) -> dict[str, float]:
    """Средние смещения сенсоров относительно S2 по парам наблюдений в один день."""
    off = {"s2": 0.0}
    pair = obs[obs.s2_ndvi.notna() & obs.landsat_ndvi.notna()]
    off["landsat"] = float((pair.landsat_ndvi - pair.s2_ndvi).mean()) if len(pair) > 30 else DEFAULT_OFFSETS["landsat"]
    pair = obs[obs.s2_ndvi.notna() & obs.modis_ndvi.notna()]
    off["modis"] = float((pair.modis_ndvi - pair.s2_ndvi).mean()) if len(pair) > 30 else DEFAULT_OFFSETS["modis"]
    off["none"] = 0.0
    return off


def estimate_offsets_by_polygon(obs: pd.DataFrame, min_pairs: int = 20,
                                global_off: dict[str, float] | None = None) -> pd.DataFrame:
    """Смещения по каждому полигону отдельно, с откатом на глобальные при нехватке пар."""
    g = global_off or estimate_offsets(obs)
    rows = []
    for pid, d in obs.groupby("anon_polygon_id"):
        rec = {"anon_polygon_id": pid, "s2": 0.0, "none": 0.0}
        for sensor in ("landsat", "modis"):
            pair = d[d.s2_ndvi.notna() & d[f"{sensor}_ndvi"].notna()]
            rec[sensor] = float((pair[f"{sensor}_ndvi"] - pair.s2_ndvi).mean()) if len(pair) >= min_pairs else g[sensor]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("anon_polygon_id")


def infer_hidden_sensor(season: pd.DataFrame, date: pd.Timestamp) -> str:
    """Угадывает сенсор скрытой точки по расписанию съёмки внутри сезона полигона.

    ``season`` — видимые строки того же полигона и того же года (с сенсорными колонками).
    """
    doy = date.dayofyear
    ls_dates = season.loc[season.landsat_ndvi.notna(), "date"]
    s2_dates = season.loc[season.s2_ndvi.notna(), "date"]
    on_modis = doy % 16 == 1
    on_ls = len(ls_dates) > 0 and bool((((date - ls_dates).dt.days % 8) == 0).any())
    on_s2 = len(s2_dates) > 0 and bool((((date - s2_dates).dt.days % 5) == 0).any())
    if on_modis and not on_ls:
        return "modis"
    if on_s2 and not on_ls and not on_modis:
        return "s2"
    if on_ls and not on_s2:
        return "landsat"
    if on_ls and on_s2:
        return "landsat"  # неоднозначно: Landsat выигрывает, он точнее опознаётся
    if on_modis:
        return "modis"
    return "landsat"


class SensorHarmonizer:
    """Линейный пересчёт между шкалами сенсоров: y_sensor = a·x_s2 + b.

    Константное смещение — грубое приближение: разница MODIS − S2 на низком NDVI +0.125,
    а на высоком −0.033; у Landsat +0.056 против +0.002. Линейная связь снимает большую часть
    этого (остаток пар одного дня: MODIS 0.108 → 0.091, Landsat 0.059 → 0.055).

    Сверху два уточнения с усадкой к общему: по году (смещение Landsat − S2 выросло с 0.015
    в 2017 до 0.05 в 2024) и по полигону (разброс между полями ±0.015).
    """

    MIN_PAIRS = 30
    YEAR_SHRINK = 100.0     # n / (n + k): сколько пар нужно, чтобы поверить годовому отклонению
    POLY_SHRINK = 40.0

    def __init__(self):
        self.global_: dict[str, tuple[float, float]] = {"s2": (1.0, 0.0), "none": (1.0, 0.0)}
        self.by_year: dict[tuple[str, int], tuple[float, float]] = {}
        self.by_poly: dict[tuple[str, str], float] = {}

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray, prior: tuple[float, float] | None = None, k: float = 0.0):
        if x.size < 3:
            return prior
        a, b = np.polyfit(x, y, 1)
        if prior is None or k <= 0:
            return float(a), float(b)
        w = x.size / (x.size + k)
        return float(w * a + (1 - w) * prior[0]), float(w * b + (1 - w) * prior[1])

    def fit(self, obs: pd.DataFrame) -> "SensorHarmonizer":
        for sensor in ("landsat", "modis"):
            col = f"{sensor}_ndvi"
            pair = obs[obs.s2_ndvi.notna() & obs[col].notna()]
            if len(pair) < self.MIN_PAIRS:
                off = DEFAULT_OFFSETS[sensor]
                self.global_[sensor] = (1.0, off)
                continue
            x, y = pair.s2_ndvi.to_numpy(float), pair[col].to_numpy(float)
            g = self._fit(x, y)
            self.global_[sensor] = g
            for yr, d in pair.groupby("year"):
                self.by_year[(sensor, int(yr))] = self._fit(
                    d.s2_ndvi.to_numpy(float), d[col].to_numpy(float), prior=g, k=self.YEAR_SHRINK)
            # по полигону корректируем только сдвиг: наклон на 20 парах не оценить
            resid = y - (g[0] * x + g[1])
            for pid, r in pd.Series(resid, index=pair.anon_polygon_id.values).groupby(level=0):
                w = r.size / (r.size + self.POLY_SHRINK)
                self.by_poly[(sensor, pid)] = float(w * r.mean())
        return self

    def coef(self, sensor: str, year=None, pid=None) -> tuple[float, float]:
        a, b = self.by_year.get((sensor, int(year)), self.global_.get(sensor, (1.0, 0.0))) \
            if year is not None else self.global_.get(sensor, (1.0, 0.0))
        if pid is not None:
            b = b + self.by_poly.get((sensor, pid), 0.0)
        return a, b

    def to_s2(self, values, sensor, year=None, pid=None):
        """Из шкалы сенсора в шкалу Sentinel-2 (векторно по массивам одинаковой длины)."""
        values = np.asarray(values, float)
        sensor = np.asarray(sensor, dtype=object)
        year = np.asarray(year) if year is not None else np.full(values.shape, None, dtype=object)
        pid = np.asarray(pid, dtype=object) if pid is not None else np.full(values.shape, None, dtype=object)
        out = np.empty_like(values)
        for i in range(values.size):
            a, b = self.coef(str(sensor[i]), year[i], pid[i])
            out[i] = (values[i] - b) / a
        return out

    def from_s2(self, values, sensor, year=None, pid=None):
        """Из шкалы Sentinel-2 в шкалу конкретного сенсора."""
        values = np.asarray(values, float)
        sensor = np.asarray(sensor, dtype=object)
        year = np.asarray(year) if year is not None else np.full(values.shape, None, dtype=object)
        pid = np.asarray(pid, dtype=object) if pid is not None else np.full(values.shape, None, dtype=object)
        out = np.empty_like(values)
        for i in range(values.size):
            a, b = self.coef(str(sensor[i]), year[i], pid[i])
            out[i] = a * values[i] + b
        return out

    def offsets(self, level: float = 0.4) -> dict[str, float]:
        """Совместимость: константные смещения на типичном уровне NDVI."""
        return {s: float(a * level + b - level) for s, (a, b) in self.global_.items()}


#: порог схожести дат, выше которого считаем, что поля снимаются одним витком
SAME_ORBIT_JACCARD = 0.5


class AcquisitionCalendar:
    """Расписание съёмки области, восстановленное из наблюдений по всем полигонам.

    Спутник снимает полосу шириной в сотни километров, поэтому в один день он «накрывает»
    сразу много полей. Если в дату гэпа Landsat снял поля, которые обычно снимаются вместе
    с нашим, значит и наш снимал Landsat.

    «Снимаются вместе» определяется по совпадению множеств дат: у полей на одном витке
    съёмки они почти одинаковы (мера Жаккара выше 0.5).
    """

    def __init__(self, jaccard_threshold: float = SAME_ORBIT_JACCARD):
        self.threshold = jaccard_threshold
        self.pids: list[str] = []
        self._index: dict[str, int] = {}
        self._by_day: dict[tuple, np.ndarray] = {}
        self._sim: dict[str, np.ndarray] = {}

    def fit(self, visible: pd.DataFrame) -> "AcquisitionCalendar":
        obs = visible[visible.src != "none"]
        self.pids = sorted(obs.anon_polygon_id.unique())
        self._index = {p: i for i, p in enumerate(self.pids)}
        n = len(self.pids)

        for (day, src), grp in obs.groupby(["date", "src"]):
            idx = [self._index[p] for p in grp.anon_polygon_id.unique()]
            self._by_day[(day, src)] = np.asarray(sorted(idx), dtype=np.int32)

        # схожесть считаем отдельно для Landsat и S2: витки у них разные
        for sensor in ("landsat", "s2"):
            sets = {p: set(d.date) for p, d in obs[obs.src == sensor].groupby("anon_polygon_id")}
            m = np.zeros((n, n), dtype=np.float32)
            for i, a in enumerate(self.pids):
                sa = sets.get(a)
                if not sa:
                    continue
                for j in range(i + 1, n):
                    sb = sets.get(self.pids[j])
                    if not sb:
                        continue
                    inter = len(sa & sb)
                    if inter:
                        v = inter / (len(sa) + len(sb) - inter)
                        m[i, j] = m[j, i] = v
            self._sim[sensor] = m
        return self

    def scores(self, pid: str, date) -> dict[str, float]:
        """Три меры на каждый сенсор: близкие поля, взвешенный счёт, просто счёт."""
        own = self._index.get(pid, -1)
        out = {}
        for sensor in SENSORS:
            idx = self._by_day.get((date, sensor))
            if idx is None or idx.size == 0:
                out[f"close_{sensor}"] = 0.0
                out[f"weighted_{sensor}"] = 0.0
                out[f"count_{sensor}"] = 0.0
                continue
            if own >= 0:
                idx = idx[idx != own]
            # для MODIS витков нет, он снимает всё подряд — берём схожесть по S2 как «соседство»
            key = "landsat" if sensor == "landsat" else "s2"
            sim = self._sim[key][own, idx] if own >= 0 and idx.size else np.zeros(idx.size)
            out[f"close_{sensor}"] = float((sim > self.threshold).sum())
            out[f"weighted_{sensor}"] = float(sim.sum())
            out[f"count_{sensor}"] = float(idx.size)
        return out

    def infer(self, pid: str, date) -> tuple[str, float]:
        """Сенсор и уверенность (доля лидера в сумме мер). ('none', 0) если календарь молчит."""
        sc = self.scores(pid, date)
        for prefix in ("close", "weighted", "count"):
            vals = {s: sc[f"{prefix}_{s}"] for s in SENSORS}
            total = sum(vals.values())
            if total > 0:
                best = max(vals, key=vals.get)
                return best, float(vals[best] / total)
        return "none", 0.0


def infer_hidden_sensors(visible: pd.DataFrame, targets: pd.DataFrame,
                         calendar: "AcquisitionCalendar | None" = None) -> pd.Series:
    """Угадывает сенсор для каждой строки ``targets``.

    Сначала спрашиваем календарь области (97.6 % попаданий), и только если он молчит —
    откатываемся на расписание самого полигона (81 %).
    """
    if calendar is None:
        calendar = AcquisitionCalendar().fit(visible)
    vis_by_key = {k: v for k, v in visible.groupby(["anon_polygon_id", "year"])}
    empty = visible.iloc[:0]
    out = []
    for pid, yr, dt in zip(targets.anon_polygon_id.values, targets.year.values, targets.date):
        src, _ = calendar.infer(pid, dt)
        if src == "none":
            src = infer_hidden_sensor(vis_by_key.get((pid, yr), empty), dt)
        out.append(src)
    return pd.Series(out, index=targets.index, name="hidden_src")


def to_s2_scale(values, sensor, offsets: dict[str, float]):
    """Приводит значения любого сенсора к шкале Sentinel-2."""
    off = pd.Series(sensor).map(offsets).fillna(0.0).values
    return np.asarray(values, dtype=float) - off


def from_s2_scale(values, sensor, offsets: dict[str, float]):
    """Обратный перевод из шкалы S2 в шкалу конкретного сенсора."""
    off = pd.Series(sensor).map(offsets).fillna(0.0).values
    return np.asarray(values, dtype=float) + off


__all__ = [
    "SENSORS", "DEFAULT_OFFSETS", "AcquisitionCalendar", "SensorHarmonizer", "estimate_offsets",
    "estimate_offsets_by_polygon", "infer_hidden_sensor", "infer_hidden_sensors",
    "to_s2_scale", "from_s2_scale",
]
