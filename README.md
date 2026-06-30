# Inventory / Demand Forecasting Assistant

MVP-сервис для прогнозирования спроса по магазинам, расчета рекомендаций по заказу и формирования управленческого отчета.

## Что умеет сервис

- загрузка исторических продаж (`CSV`);
- запуск прогноза по загруженному датасету;
- запуск `random-val` инференса на данных Rossmann (`models/hgb_full.joblib` + preprocessing + postprocessing);
- формирование рекомендаций по заказу (`status`, `priority`, `action`, `reason`);
- генерация итогового отчета (`executive summary`, `kpi`, `main insights`, `store-level actions`);
- UI с аутентификацией, дашбордом и историей запусков.

## Архитектура выполнения прогнозов

`POST /forecast/run` работает неблокирующе:

1. API создает `ForecastRun` со статусом `queued` и сразу отвечает `202`.
2. Отдельный процесс `forecast-worker` забирает задачи из очереди (таблица `forecastrun`) и переводит их в `running`.
3. После расчета статус становится `completed` (или `failed` при ошибке).

Горизонтальное масштабирование:

- можно поднимать несколько воркеров параллельно:
  - `docker compose up -d --scale forecast-worker=3`

## Быстрый старт (Docker Compose, рекомендуется)

1. Создай `.env`:
   - `cp .env.example .env`
2. Запусти сервисы:
   - `docker compose up --build -d`
3. Проверь доступность:
   - Swagger: `http://localhost:8000/docs`
   - UI: `http://localhost:8000/login`
4. Остановка:
   - `docker compose down`

## Локальный запуск (без Docker для API/worker)

1. Подними только PostgreSQL:
   - `docker compose up -d db`
2. Примени миграции:
   - `uv run alembic upgrade head`
3. Запусти API:
   - `uv run python main.py`
4. Запусти worker в отдельном терминале:
   - `uv run python -m app.workers.forecast_worker`

## Дефолтный пользователь

При инициализации БД создается:

- login: `admin`
- password: `admin`

## Основные API endpoints

- `POST /datasets/upload`
- `POST /forecast/run` (асинхронная постановка в очередь)
- `POST /forecast/run/random-val`
- `GET /forecast/{id}`
- `GET /forecast/run/{id}/status`
- `GET /recommendations/{id}`
- `GET /reports/{id}`
- `GET /models`
- `GET /health`

## Метрики и интерпретация

Сервис считает несколько уровней метрик:

- **Прогнозные** (по горизонту): `total_demand`, `avg_daily_demand`, `max_daily_demand`.
- **Сравнение с прошлым периодом**: `change_vs_previous_period`.
- **Стабильность/пики**: `demand_spike_index`, `forecast_volatility`.
- **Планируемый спрос для заказа**: `expected_demand` (спрос с буфером риска).
- **Дашборд KPI**:
  - `expected_demand` — суммарный ожидаемый спрос;
  - `recommended_order` — суммарный рекомендованный объем заказа;
  - `priority` — максимальный приоритет среди магазинов;
  - `demand_vs_usual` — среднее отклонение спроса от baseline;
  - `stores_requiring_action` — количество магазинов, где требуется действие.

Подробная расшифровка логики и формул: `docs/service_guide.md`.

## Документация

- обзор схемы БД: `docs/db_schema.md`
- эксплуатационная и продуктовая документация: `docs/service_guide.md`

## Тесты

- запуск:
  - `DATABASE_URL=sqlite:///./test_suite.db uv run pytest -q --cov=app --cov-report=term-missing`

