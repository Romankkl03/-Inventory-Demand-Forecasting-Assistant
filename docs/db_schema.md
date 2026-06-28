# Database Schema (PostgreSQL)

This schema is defined by SQLModel entities in `app/models/entities.py` and applied via Alembic migration `alembic/versions/c873a1a8d077_initial_schema.py`.

## Core Forecasting Tables

- `user`: service users.
- `store`: stores for demand forecasting.
- `salesrecord`: historical daily sales by store.
- `dataset`: uploaded datasets.
- `modelversion`: trained model versions with metrics JSON.
- `forecastrun`: forecasting jobs.
- `forecast`: per-store, per-date predicted sales.
- `recommendation`: recommendation output derived from forecasts.
- `report`: aggregated report for forecast runs.

## Subscription Tables

- `tariffplan`: available service plans.
- `subscription`: user subscriptions to tariff plans.
- `payment`: subscription payment history.
- `usagelimit`: limits attached to tariff plans.
- `subscriptionfeature`: plan feature flags.

## Main Relationships

- `dataset.uploaded_by -> user.id`
- `salesrecord.store_id -> store.id`
- `forecastrun.dataset_id -> dataset.id`
- `forecastrun.model_version_id -> modelversion.id`
- `forecastrun.created_by -> user.id`
- `forecast.forecast_run_id -> forecastrun.id`
- `forecast.store_id -> store.id`
- `recommendation.forecast_run_id -> forecastrun.id`
- `recommendation.store_id -> store.id`
- `report.forecast_run_id -> forecastrun.id`
- `report.created_by -> user.id`
- `subscription.user_id -> user.id`
- `subscription.tariff_plan_id -> tariffplan.id`
- `payment.subscription_id -> subscription.id`
- `payment.user_id -> user.id`
- `usagelimit.tariff_plan_id -> tariffplan.id`
- `subscriptionfeature.tariff_plan_id -> tariffplan.id`
