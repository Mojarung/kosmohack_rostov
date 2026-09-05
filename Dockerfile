# Образ веб-сервиса и batch-инференса. Сборка: docker build -t kosmohack .
# Запуск сервиса: docker run -p 8000:8000 kosmohack
# Batch-инференс: docker run -v $PWD/data:/app/data -v $PWD/out:/app/out kosmohack \
#   uv run python -m gapfill.predict --input data/test_dataset.csv --output out/submission.csv
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONIOENCODING=utf-8 UV_TORCH_BACKEND=cpu

# Зависимости отдельно от кода, чтобы слой кэшировался
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group ml --group geo --group service

COPY . .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
