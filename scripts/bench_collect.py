"""Замер скорости сбора одного сезона: где тратится время и как влияет число потоков.

Запуск: PYTHONPATH=. uv run --no-sync python scripts/bench_collect.py [--source MODIS|Landsat|S2|S2ES] [--year 2024]
        [--workers 10] [--multiplex YES|NO]
Печатает время поиска в каталоге, время загрузки и число обрывов чтения (по логам odc/rasterio).
Полигон — квадрат 1×1 км под Ростовом, как у типичного поля пользователя.
"""

from __future__ import annotations

import argparse
import logging
import time

import dask
from odc.stac import configure_rio
from shapely.geometry import box

from service import collect


class _Counter(logging.Handler):
    """Считает предупреждения об обрывах чтения, которые odc-stac молча проглатывает."""

    def __init__(self) -> None:
        super().__init__()
        self.failures = 0
        self.retries = 0

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        if "Ignoring read failure" in text:
            self.failures += 1
        if "Retrying again" in text:
            self.retries += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="MODIS", choices=["MODIS", "Landsat", "S2", "S2ES"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=10, help="потоков dask на один вызов stac_load")
    parser.add_argument("--multiplex", default="YES", choices=["YES", "NO"])
    args = parser.parse_args()

    counter = _Counter()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().addHandler(counter)
    collect.configure_gdal()
    if args.multiplex == "NO":
        # тот же набор, что в configure_gdal, но без HTTP/2 и мультиплексирования
        configure_rio(cloud_defaults=True, GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MULTIPLEX="NO",
                      GDAL_HTTP_VERSION="1.1", VSI_CACHE="TRUE", GDAL_HTTP_TIMEOUT=60,
                      GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1)
    dask.config.set(num_workers=args.workers)

    # разделяем время поиска в каталоге и время загрузки снимков
    timings = {"load": 0.0}
    original = collect._load_parallel

    def timed_load(*a, **kw):
        t = time.time()
        try:
            return original(*a, **kw)
        finally:
            timings["load"] += time.time() - t
    collect._load_parallel = timed_load

    geom = box(39.70, 47.20, 39.713, 47.209)    # ~1 км × 1 км
    # S2 — как в сервисе (Planetary Computer, при отказе Earth Search); S2ES — принудительно Earth Search для сравнения
    fn = {"MODIS": collect._modis_year, "Landsat": collect._landsat_year, "S2": collect._s2_year,
          "S2ES": lambda g, y: collect._s2_year_from(collect.EARTH_SEARCH, g, y)}[args.source]
    t0 = time.time()
    frame = fn(geom, args.year)
    total = time.time() - t0
    print(f"{args.source} {args.year}: {len(frame)} сцен за {total:.0f} с (поиск {total - timings['load']:.0f} с, "
          f"загрузка {timings['load']:.0f} с); workers={args.workers}, multiplex={args.multiplex}; "
          f"обрывов чтения {counter.failures}, повторов {counter.retries}")


if __name__ == "__main__":
    main()
