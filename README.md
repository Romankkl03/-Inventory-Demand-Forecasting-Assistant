# Inventory / Demand Forecasting Assistant

Python project for inventory demand forecasting, model experimentation, and serving forecast results.

## Database setup (PostgreSQL)

1. Create env from template:
   - `cp .env.example .env`
2. Start PostgreSQL:
   - `docker compose up -d db`
3. Apply migrations:
   - `uv run alembic upgrade head`

Application code lives in `app/`. DB connection is configured in `app/database/config.py`.
Schema overview is in `docs/db_schema.md`.

## Run API

- `uv run python main.py`
- Swagger docs: `http://localhost:8000/docs`
- UI:
  - `http://localhost:8000/login`
  - `http://localhost:8000/signup`
  - `http://localhost:8000/dashboard`

## Run API in Docker Compose

1. Create local env:
   - `cp .env.example .env`
2. Start services:
   - `docker compose up --build -d`
3. Open Swagger:
   - `http://localhost:8000/docs`
4. Stop services:
   - `docker compose down`

Main endpoints:

- `POST /datasets/upload`
- `POST /forecast/run`
- `POST /forecast/run/random-val` (random row inference from Rossmann validation split using `models/hgb_full.joblib`)
- `GET /forecast/{id}`
- `GET /forecast/run/{id}/status`
- `GET /recommendations/{id}`
- `GET /reports/{id}`
- `GET /models`
- `GET /health`

## Project layout

```text
app/
  models/           # SQLModel entities (DB schema)
  database/         # DB config, session, init
  forecasting/      # ML pipeline (data, preprocessing, metrics, postprocessing)
  routes/           # API routes (placeholder)
  services/         # business logic (placeholder)
```

Legacy code in `src/` is kept temporarily during migration; use `app/` for new work.

