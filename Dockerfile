# Образ всего продукта: интерфейс (React + Vite) собирается на первом этапе,
# бэкенд (FastAPI + модели) — на втором, готовая статика попадает внутрь образа.
#
# Одна команда для запуска всего:      docker compose up --build
# Или вручную:                         docker build -t kosmohack . && docker run -p 8000:8000 kosmohack
# Batch-инференс (задача 1) в контейнере:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/out:/app/out" kosmohack \
#     uv run --no-sync python -m gapfill.predict_improved --output artifacts/submission.csv --device cpu

# --- этап 1: сборка интерфейса ---
FROM node:24-alpine AS web

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- этап 2: сервис ---
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONIOENCODING=utf-8 UV_TORCH_BACKEND=cpu PYTHONUNBUFFERED=1

# Системные библиотеки, которых нет в slim-образе:
#   libgomp1  — OpenMP для LightGBM;
#   libexpat1 — нужна rasterio (через GDAL внутри колеса), без неё падает весь сбор данных,
#               а вместе с ним и проверка окружения при старте сервиса.
RUN apt-get update     && apt-get install -y --no-install-recommends libgomp1 libexpat1     && rm -rf /var/lib/apt/lists/*

# Зависимости отдельно от кода, чтобы слой кэшировался между сборками.
# В образ идут только группы, нужные для работы: geo (сбор данных), serve (FastAPI),
# infer (LightGBM, scikit-learn). Группа ml с CatBoost, XGBoost, Optuna, SHAP и Numba нужна
# только для экспериментов: на Linux она тянет nvidia-* и раздувает образ на десятки гигабайт.
# --no-cache: иначе загруженные колёса остаются в слое рядом с распакованным venv.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache --group infer --group geo --group serve

# torch ставится отдельно из индекса CPU-сборок: в общем lock-файле линуксовый torch тянет пакеты
# nvidia-* (несколько гигабайт), а для инференса SeasonNet из models/ достаточно CPU.
# Полная группа dl (Chronos, PyPOTS) нужна только для экспериментов и в образ не входит.
RUN uv pip install --no-cache "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu

COPY . .
COPY --from=web /web/dist ./web/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/summary', timeout=4)"
CMD ["uv", "run", "--no-sync", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
