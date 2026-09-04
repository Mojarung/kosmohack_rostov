"""Пути, константы и параметры визуализации для разведочного анализа (EDA)."""

from pathlib import Path

# Корень репозитория: eda/config.py -> eda -> корень
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"
TEST_PATH = DATA_DIR / "test_dataset.csv"

REPORT_DIR = ROOT / "reports" / "eda"
FIG_DIR = REPORT_DIR / "figures"
SUMMARY_PATH = REPORT_DIR / "summary.json"

# Ключ строки: одна строка = один полигон в одну дату
KEY = ["anon_polygon_id", "date"]
TARGET = "primary_ndvi"

# Сенсорные колонки в порядке приоритета формирования primary_ndvi (проверено по данным)
SENSOR_NDVI = ["s2_ndvi", "landsat_ndvi", "modis_ndvi"]
SENSOR_ALL = [
    "s2_ndvi", "s2_evi", "s2_ndwi",
    "landsat_ndvi", "landsat_evi", "landsat_ndwi",
    "modis_ndvi", "modis_evi",
]
WEATHER = ["era5_temp_c", "era5_precip_mm"]
CLIMATOLOGY = ["ndvi_climatology_mean", "ndvi_climatology_std", "n_reference_years"]

# Порядок категорий для единообразных графиков
CROP_ORDER = ["озимая пшеница", "зерновые", "подсолнечник", "пастбища/зерновые"]
STATUS_ORDER = ["Штатное развитие", "Угнетение биомассы", "Критическая аномалия"]
STATUS_COLORS = {
    "Штатное развитие": "#3a9d5d",
    "Угнетение биомассы": "#e8a33d",
    "Критическая аномалия": "#c8423f",
}
SENSOR_COLORS = {"s2_ndvi": "#2b6cb0", "landsat_ndvi": "#805ad5", "modis_ndvi": "#dd6b20"}

# Пороги Z-score из ТЗ
Z_DEPRESSION = -1.0
Z_CRITICAL = -2.0

# Порог метрики: RMSE = 0.10 даёт GapScore = 0
RMSE_THRESHOLD = 0.10
GAP_SCORE_MAX = 30

# Сезонное окно наблюдений (день года), как в данных
DOY_MIN, DOY_MAX = 91, 304

RANDOM_SEED = 42
