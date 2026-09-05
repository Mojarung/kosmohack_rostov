"""Чтение отражательных каналов только для дней с достаточной долей чистых пикселей."""

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def load_clear_days(lazy, inside, min_pixels, min_valid_share, clear_codes, bands=("red", "nir", "scl")):
    """SCL даёт верхнюю границу пригодных пикселей; окончательный NDVI считает сборщик.

    Выбор из общего плана сохраняет сетку, даты и все версии дневной мозаики.
    Чанк времени должен быть равен одному дню, чтобы исключить ненужное чтение.
    `bands` — имена каналов (red, nir, scl) в этом каталоге: у Planetary Computer это B04/B08/SCL.
    """
    red, nir, scl_band = bands
    started = time.perf_counter()
    scl = lazy[scl_band].compute()
    mask_seconds = time.perf_counter() - started
    n_inside = int(inside.sum())
    clear = np.isin(scl.values, clear_codes) & inside
    counts = clear.sum(axis=(-2, -1))
    keep = np.flatnonzero((counts / max(n_inside, 1) >= min_valid_share) & (n_inside >= min_pixels))
    started = time.perf_counter()
    data = lazy[[red, nir]].isel(time=keep).compute()
    data[scl_band] = scl.isel(time=keep)
    data.attrs.update(s2_days_total=len(scl.time), s2_days_kept=len(keep),
                      s2_mask_seconds=mask_seconds, s2_reflectance_seconds=time.perf_counter() - started)
    log.info("S2: SCL %.1f с, red/nir %.1f с; оставлено %d из %d дней",
             mask_seconds, data.attrs["s2_reflectance_seconds"], len(keep), len(scl.time))
    return data
