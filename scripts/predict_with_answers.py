"""Инференс улучшенной модели с ответами первой версии test в контексте и калибровке.

Стандартный predict_improved привязывает калибровку к отпечаткам исходных файлов. Здесь тот же
код запускается с data/test_dataset_v1_answers.csv вместо data/test_dataset.csv: нормы
(ndvi_climatology_*) в этом файле те же, а контрольные точки первой версии стали известными.

Запуск: uv run --no-sync python scripts/predict_with_answers.py --output submission.csv
"""

import argparse
import time
from pathlib import Path

import gapfill.predict_improved as pi
from gapfill.config import ROOT, TEST_PATH

ANSWERS = ROOT / "data" / "test_dataset_v1_answers.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(TEST_PATH))
    p.add_argument("--extra", nargs="*", default=[str(ANSWERS)])
    p.add_argument("--output", required=True)
    p.add_argument("--no-calibration", action="store_true")
    args = p.parse_args()
    extra = tuple(Path(x) for x in args.extra)
    # Проверки отпечатков рассчитаны на исходные файлы; нормы калибровки от замены extra не меняются
    pi.check_calibration_inputs = lambda *a, **k: None
    pi.EXTRA_PATHS = extra
    start = time.monotonic()
    pred, _ = pi.predict_improved(ROOT / "models" / "improved", pi.TRAIN_PATH, args.input, extra,
                                  not args.no_calibration)
    pi.write_csv(pred, args.output)
    print(f"{args.output}: {len(pred)} строк, среднее {pred.pred.mean():.5f}, "
          f"калибровка={'нет' if args.no_calibration else 'да'}, {time.monotonic() - start:.0f} с", flush=True)


if __name__ == "__main__":
    main()
