"""Константы и пути для пайплайна восстановления primary_ndvi."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"
TEST_PATH = DATA_DIR / "test_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"          # модели, признаки, промежуточные файлы (не в git)
SUBMISSION_PATH = ROOT / "submission.csv"

TARGET = "primary_ndvi"
EPOCH = "2000-01-01"                        # начало отсчёта day_num

# Сенсоры в порядке приоритета формирования primary_ndvi
SENSORS = ["s2", "landsat", "modis"]
SENSOR_CODE = {"s2": 0, "landsat": 1, "modis": 2}
SENSOR_NDVI = {"s2": "s2_ndvi", "landsat": "landsat_ndvi", "modis": "modis_ndvi"}
INDEX_COLS = ["s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi", "landsat_ndwi",
              "modis_ndvi", "modis_evi"]
WEATHER_COLS = ["era5_temp_c", "era5_precip_mm"]

# Глобальные смещения сенсоров относительно Sentinel-2 (из EDA, совместные наблюдения в один день)
SENSOR_OFFSET = {"s2": 0.0, "landsat": 0.037, "modis": 0.083}

# Циклы съёмки: период в днях; MODIS живёт на фиксированной сетке дней года (MOD13Q1)
CYCLE_PERIOD = {"s2": 5, "landsat": 8}
MODIS_GRID_DOY = tuple(range(1, 366, 16))   # 1, 17, 33, …, 97, 113, …, 289, 305, …
MODIS_WINDOW = 15                            # композит MOD13Q1 датирован началом окна [d, d+15]

# Доля контрольных точек среди исходно известных в test (3 112 / 20 753)
GAP_SHARE = 0.15

# Полосы ядра локальной линейной регрессии (дни)
BANDWIDTHS = (6.0, 15.0)
KERNEL_CUTOFF = 3.0                          # обрезка ядра в полосах

# Число соседей с каждой стороны для признаков
N_NEIGHBORS = 3
MAX_NEIGHBOR_DAYS = 60                       # соседи дальше — считаются отсутствующими

CROP_CODES = {"озимая пшеница": 0, "зерновые": 1, "подсолнечник": 2, "пастбища/зерновые": 3}

RANDOM_SEED = 42
