FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY app /app/app
COPY main.py /app/main.py
COPY models /app/models
COPY data /app/data

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
