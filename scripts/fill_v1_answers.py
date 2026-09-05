"""Подставляет ответы организаторов к первой версии test в её контрольные точки.

Организаторы выложили private_test_ground_truth.csv (3 112 точек первой версии test). Эти точки
становятся обычными известными наблюдениями: они идут в контекст модели (соседи по времени,
«шум дня» других полигонов) и в калибровку историческими агрегатами. Сенсор точки назначается
правилом из EDA (S2 по остатку %5 в том же полигоне-году, иначе Landsat по %8, иначе MODIS по сетке).

Запуск: uv run --no-sync python scripts/fill_v1_answers.py
Выход: data/test_dataset_v1_answers.csv (все 3 112 точек известны) и два файла для проверки A/B
       в artifacts/gt_ab/: половина A известна + половина B скрыта, и исходный v1 (эталон).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "test_dataset.csv"
V2 = ROOT / "data" / "test_features_new.csv"
TRAIN = ROOT / "data" / "train_dataset.csv"
GT = ROOT / "data" / "private_test_ground_truth.csv"
OUT = ROOT / "data" / "test_dataset_v1_answers.csv"
AB_DIR = ROOT / "artifacts" / "gt_ab"
KEYS = ["anon_polygon_id", "date"]
EPOCH = pd.Timestamp("2000-01-01")


def _with_calendar(df):
    """Номер дня от эпохи, год и день года для правила сенсора."""
    date = pd.to_datetime(df["date"])
    return df.assign(day_num=(date - EPOCH).dt.days, year=date.dt.year, cal_doy=date.dt.dayofyear)


def _cycle_counts(reference, col, period):
    """Сколько наблюдений сенсора в полигоне-году имеют данный остаток дня по периоду."""
    d = reference.loc[reference[col].notna()].assign(res=lambda x: x["day_num"] % period)
    return d.groupby(["anon_polygon_id", "year", "res"]).size().rename("cnt").reset_index()


def _count_for(points, counts, period):
    keyed = points.assign(res=points["day_num"] % period)
    merged = keyed[["anon_polygon_id", "year", "res"]].merge(counts, on=["anon_polygon_id", "year", "res"], how="left")
    return merged["cnt"].fillna(0).to_numpy()


def guess_sensor(points, reference):
    """Правило из EDA (92 % на train): s2 → landsat → modis по сетке; иначе s2 как самый частый."""
    cnt_s2 = _count_for(points, _cycle_counts(reference, "s2_ndvi", 5), 5)
    cnt_ls = _count_for(points, _cycle_counts(reference, "landsat_ndvi", 8), 8)
    modis_days = set(reference.loc[reference["modis_ndvi"].notna(), "cal_doy"].unique())
    on_grid = points["cal_doy"].isin(modis_days).to_numpy()
    return np.select([cnt_s2 > 0, cnt_ls > 0, on_grid], ["s2_ndvi", "landsat_ndvi", "modis_ndvi"], default="s2_ndvi")


def fill(v1, gt, reference, keys_to_fill):
    """Возвращает копию v1, где выбранные контрольные точки стали известными наблюдениями."""
    out = v1.copy()
    is_gap = out["is_synthetic_gap"].astype(str).str.lower().eq("true")
    sel = is_gap & pd.MultiIndex.from_frame(out[KEYS]).isin(keys_to_fill)
    rows = out.loc[sel].merge(gt, on=KEYS, how="left")
    assert rows["primary_ndvi_true"].notna().all(), "нет ответа для части точек"
    sensor_col = guess_sensor(rows, reference)
    values = rows["primary_ndvi_true"].to_numpy()
    out.loc[sel, "primary_ndvi"] = values
    out.loc[sel, "is_synthetic_gap"] = False
    for col in ("s2_ndvi", "landsat_ndvi", "modis_ndvi"):
        idx = out.index[sel][sensor_col == col]
        out.loc[idx, col] = values[sensor_col == col]
    print(f"заполнено {int(sel.sum())} точек; сенсоры: {pd.Series(sensor_col).value_counts().to_dict()}")
    return out


def main():
    gt = pd.read_csv(GT)
    v1 = _with_calendar(pd.read_csv(V1, low_memory=False))
    reference = pd.concat([v1, _with_calendar(pd.read_csv(V2, low_memory=False)),
                           _with_calendar(pd.read_csv(TRAIN, low_memory=False))], ignore_index=True)
    drop = ["day_num", "year", "cal_doy"]
    all_keys = pd.MultiIndex.from_frame(gt[KEYS])
    fill(v1, gt, reference, all_keys).drop(columns=drop).to_csv(OUT, index=False, encoding="utf-8")
    print("записан", OUT)
    # A/B: половина A становится известной, половина B остаётся скрытой и оценивается по ответам
    rng = np.random.default_rng(0)
    half_a = rng.random(len(gt)) < 0.5
    keys_a = pd.MultiIndex.from_frame(gt.loc[half_a, KEYS])
    AB_DIR.mkdir(parents=True, exist_ok=True)
    fill(v1, gt, reference, keys_a).drop(columns=drop).to_csv(AB_DIR / "v1_half_a_known.csv", index=False, encoding="utf-8")
    gt.loc[~half_a].to_csv(AB_DIR / "truth_half_b.csv", index=False, encoding="utf-8")
    print("A/B файлы записаны в", AB_DIR)


if __name__ == "__main__":
    main()
