# Образ сервиса: GDAL идёт бинарным колесом внутри rasterio, системный GDAL не нужен
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    AWS_NO_SIGN_REQUEST=YES

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[web]"

COPY scripts ./scripts
COPY web ./web
COPY data ./data

# модель обучается на этапе сборки, чтобы контейнер стартовал готовым к работе
RUN python scripts/train.py

EXPOSE 8000
CMD ["uvicorn", "ndvi.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
