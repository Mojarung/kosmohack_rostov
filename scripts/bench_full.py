"""Замер полного сбора (все источники, N сезонов) при разных лимитах одновременных загрузок.

Запуск: PYTHONPATH=. uv run --no-sync python scripts/bench_full.py --years 2019 2026 --workers 8 --dask 10 --limit 0
  --workers  сезонов одного источника, загружаемых одновременно (как WORKERS в service.collect)
  --dask     потоков dask на один вызов stac_load (по умолчанию в коде — число CPU, 10)
  --limit    общий лимит одновременных загрузок снимков на все источники; 0 — без лимита (как сейчас)
Печатает время по источникам и сезонам, число сцен и число обрывов чтения. Код сервиса не меняет:
лимит навешивается на collect._load_parallel только в этом процессе.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import dask
from shapely.geometry import box

from service import collect
from service.progress import Progress


class _Counter(logging.Handler):
    """Считает обрывы чтения и повторы, которые odc-stac и GDAL проглатывают молча."""

    def __init__(self) -> None:
        super().__init__()
        self.failures = 0
        self.retries = 0

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        self.failures += "Ignoring read failure" in text
        self.retries += "Retrying again" in text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs=2, type=int, default=[2019, 2026])
    parser.add_argument("--workers", type=int, default=collect.WORKERS)
    parser.add_argument("--dask", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    counter = _Counter()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in ("rasterio", "odc", "urllib3", "botocore"):
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(counter)
    logging.getLogger("odc.loader._rio").setLevel(logging.WARNING)
    collect.configure_gdal()
    dask.config.set(num_workers=args.dask)
    collect.WORKERS = args.workers

    if args.limit > 0:
        gate = threading.Semaphore(args.limit)
        original = collect._load_parallel

        def limited(*a, **kw):
            with gate:
                return original(*a, **kw)
        collect._load_parallel = limited

    geom = box(39.70, 47.20, 39.713, 47.209)    # ~1 км × 1 км под Ростовом
    years = range(args.years[0], args.years[1] + 1)
    t0 = time.time()
    frames, _notes, warnings = collect.collect_all(geom, years, "bench", Progress(None))
    total = time.time() - t0
    print(f"\nИТОГ: {total:.0f} с; workers={args.workers}, dask={args.dask}, limit={args.limit or 'нет'}; "
          f"обрывов чтения {counter.failures}, повторов {counter.retries}")
    for label, frame in frames.items():
        print(f"  {label}: {len(frame)} сцен, предупреждений {len(frame.attrs.get('warnings', []))}")
    for w in warnings:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
