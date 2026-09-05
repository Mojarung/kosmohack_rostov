# Образ всего продукта: интерфейс (React + Vite) собирается на первом этапе,
# бэкенд (FastAPI + модели) — на втором, готовая статика попадает внутрь образа.
#
# Одна команда для запуска всего:      docker compose up --build
# Или вручную:                         docker build -t kosmohack . && docker run -p 8000:8000 kosmohack
# Batch-инференс (задача 1) в контейнере:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/out:/app/out" kosmohack \
#     uv run python -m gapfill.predict_saved --input data/test_features_new.csv --output out/submission.csv

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

# Зависимости отдельно от кода, чтобы слой кэшировался между сборками.
# Группа dl (torch, CPU) нужна для инференса нейросети из models/; сервису самому она не требуется.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group ml --group geo --group service --group dl

COPY . .
COPY --from=web /web/dist ./web/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/summary', timeout=4)"
CMD ["uv", "run", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
