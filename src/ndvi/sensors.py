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
    "SENSORS", "DEFAULT_OFFSETS", "AcquisitionCalendar", "estimate_offsets",
    "estimate_offsets_by_polygon", "infer_hidden_sensor", "infer_hidden_sensors",
    "to_s2_scale", "from_s2_scale",
]
