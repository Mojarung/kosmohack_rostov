"""Сравнение первого чтения S2 в отдельных процессах на фиксированном каталоге."""

import argparse
import json
import logging
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import pystac_client
from shapely.geometry import box

from service import collect
from service.s2_cloud_filter import load_clear_days

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "s2-cloud-filter"
GEOM = box(39.70, 47.20, 39.713, 47.209)


def prepare(period):
    """Каталог общий для всех прогонов, его получение не входит во время чтения."""
    client = pystac_client.Client.open(collect.EARTH_SEARCH)
    items = collect.public_s2_items(list(client.search(collections=["sentinel-2-l2a"],
        intersects=GEOM.__geo_interface__, datetime=period,
        query={"eo:cloud_cover": {"lt": collect.MAX_CLOUD_COVER}}).items()))
    (OUT / "items.json").write_text(json.dumps([i.to_dict() for i in items]), encoding="utf-8")
    print(json.dumps({"items": len(items), "period": period}), flush=True)


def run(mode, name):
    """Сохраняет массивы и итоговые наблюдения для точного сравнения после замеров."""
    items = [pystac.Item.from_dict(i) for i in json.loads((OUT / "items.json").read_text(encoding="utf-8"))]
    collect.configure_gdal()
    started = time.perf_counter()
    lazy = collect._plan_load(items, ["red", "nir", "scl"], GEOM, 20)
    inside = collect._polygon_mask(lazy, GEOM)
    if mode == "filter":
        data = load_clear_days(lazy, inside, collect.MIN_PIXELS, collect.MIN_VALID_SHARE, collect.S2_CLEAR_SCL)
    else:
        data = lazy.compute()
    elapsed = time.perf_counter() - started
    rows = []
    for t in data.time.values:
        frame = data.sel(time=t)
        ndvi, share, _ = collect._mean_index(frame.nir.values.astype(float), frame.red.values.astype(float),
            np.isin(frame.scl.values, collect.S2_CLEAR_SCL), inside)
        if np.isfinite(ndvi):
            rows.append({"date": str(pd.Timestamp(t).normalize()), "ndvi": ndvi, "share": share})
    result = {"mode": mode, "seconds": elapsed, "items": len(items), "days": len(data.time),
              "observations": rows, **data.attrs}
    np.savez_compressed(OUT / f"{name}.npz", time=data.time.values,
                        **{b: data[b].values for b in ("red", "nir", "scl")})
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, default=str), flush=True)


def compare(pair):
    """Сравнивает реальные загруженные пиксели и принятые наблюдения двух прогонов."""
    baseline = json.loads((OUT / f"baseline-{pair}.json").read_text(encoding="utf-8"))
    filtered = json.loads((OUT / f"filter-{pair}.json").read_text(encoding="utf-8"))
    with np.load(OUT / f"baseline-{pair}.npz") as before, np.load(OUT / f"filter-{pair}.npz") as after:
        selected = np.flatnonzero(np.isin(before["time"], after["time"]))
        equal = {band: np.array_equal(before[band][selected], after[band], equal_nan=True)
                 for band in ("red", "nir", "scl")}
        timestamps_equal = np.array_equal(before["time"][selected], after["time"])
    result = {"baseline_seconds": baseline["seconds"], "filter_seconds": filtered["seconds"],
              "days_total": baseline["days"], "days_kept": filtered["days"],
              "observations_equal": baseline["observations"] == filtered["observations"],
              "pixels_equal": equal, "timestamps_equal": timestamps_equal}
    (OUT / f"comparison-{pair}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "baseline", "filter", "compare"])
    parser.add_argument("--name")
    parser.add_argument("--period", default="2021-04-01/2021-10-30")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    signal.alarm(180)
    if args.mode == "prepare":
        prepare(args.period)
    elif args.mode == "compare":
        compare(args.name or "1")
    else:
        run(args.mode, args.name or args.mode)
