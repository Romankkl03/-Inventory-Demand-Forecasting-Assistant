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

