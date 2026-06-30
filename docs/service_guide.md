# Service Guide

## 1) Сценарий пользователя

1. Пользователь логинится/регистрируется в UI.
2. Загружает CSV с продажами или запускает `Random Example from Val`.
3. Создает запуск прогноза.
4. Получает:
   - график истории + прогноза;
   - action table по заказам;
   - итоговый management summary;
   - шаблон письма поставщику.

## 2) Асинхронное выполнение и масштабирование

### Как работает очередь запусков

- API endpoint `POST /forecast/run` не считает прогноз в HTTP-запросе.
- Вместо этого создается запись `ForecastRun` со статусом `queued`.
- Worker (`app.workers.forecast_worker`) циклически:
  1. берет следующую задачу `queued`;
  2. переводит ее в `running`;
  3. выполняет расчет и сохраняет `Forecast`;
  4. переводит в `completed` или `failed`.

### Почему можно запускать несколько воркеров

Выбор задач идет через блокировку `FOR UPDATE SKIP LOCKED` (в SQLAlchemy: `with_for_update(skip_locked=True)`), поэтому разные worker-процессы не подхватывают один и тот же `ForecastRun`.

### Как масштабировать

- через Docker Compose:
  - `docker compose up -d --scale forecast-worker=3`
- через параметр polling:
  - `FORECAST_WORKER_POLL_SEC` (по умолчанию `1` сек).

## 3) Основная функциональность по модулям

- `app/routes/*`: REST API + UI endpoints.
- `app/services/forecasting_service.py`:
  - создание run;
  - очередь/исполнение;
  - random-val inference.
- `app/services/recommendation_engine/aggregator.py`:
  - расчет агрегатных признаков спроса.
- `app/services/recommendation_engine/rules.py`:
  - deterministic rule-based рекомендация.
- `app/services/recommendation_service.py`:
  - формирование action table.
- `app/services/report_service.py`:
  - management summary + KPI блок.

## 4) Метрики прогноза и рекомендаций

Ниже — практический смысл метрик и как они считаются.

### 4.1 Прогнозные агрегаты (aggregator)

- `horizon_days`: число уникальных дат в горизонте.
- `total_demand`: сумма `predicted_sales` за горизонт.
- `avg_daily_demand`: средний дневной спрос.
- `max_daily_demand`: максимальный дневной спрос.
- `change_vs_previous_period`:
  - `(total_demand - previous_period_total) / previous_period_total`.
- `demand_spike_index`:
  - `max_daily_demand / avg_daily_demand`.
- `forecast_volatility`:
  - `std(predicted_sales) / avg_daily_demand`.
- `high_demand_regime=True`, если выполняется хотя бы одно:
  - `change_vs_previous_period >= 0.20`;
  - `demand_spike_index >= 1.60`;
  - `forecast_volatility >= 0.35`.
- `expected_demand`:
  - `total_demand * (1 + risk_buffer + regime_buffer)`, где:
    - `risk_buffer = min(0.25, max(0, forecast_volatility * 0.5))`;
    - `regime_buffer = 0.10` при `high_demand_regime`, иначе `0.03`.

### 4.2 Рекомендации (rules)

- `recommended_order`:
  - рассчитывается на окно `lead_time` + safety stock - текущая позиция запаса.
- `risk_level` (`low`, `medium`, `high`):
  - на основе `risk_score`, который учитывает:
    - high demand regime;
    - рост к прошлому периоду;
    - spike;
    - volatility;
    - low inventory cover.
- `reason_flags`:
  - машинные теги причин, потом преобразуются в человекочитаемые `reason_tags`.

### 4.3 Action table (UI/API)

Колонки:

- `Status`: `Increase order` / `Maintain order` / `Reduce order`.
- `Demand vs baseline`: `%` отклонение от предыдущего периода на горизонте.
- `Priority`: `Low` / `Medium` / `High`.
- `Reason`: объединенные human-readable причины.
- `Action`: конкретное действие (`Urgent supplier reorder`, `Place replenishment order`, и т.д.).

## 5) KPI и отчет

`ReportResponse` возвращает:

- `executive_summary`:
  - `processed_stores`;
  - `total_expected_demand`;
  - `total_recommended_order`;
  - `high_risk_stores`;
  - `main_conclusion`.
- `kpis`:
  - `expected_demand`, `recommended_order`, `priority`,
    `demand_vs_usual`, `stores_requiring_action`.
- `main_insights`: ключевые тезисы для менеджмента.
- `store_level_actions`: действия по каждому магазину.

Если настроен `VLLM_BASE_URL`, summary может быть дообогащен LLM-генерацией.

## 6) Запуск и проверка

### Docker

1. `cp .env.example .env`
2. `docker compose up --build -d`
3. `docker compose up -d --scale forecast-worker=3` (опционально)
4. Проверка:
   - `GET /health`
   - `GET /forecast/run/{id}/status`

### Локально

1. `docker compose up -d db`
2. `uv run alembic upgrade head`
3. `uv run python main.py`
4. `uv run python -m app.workers.forecast_worker`

## 7) Ограничения MVP

- В MVP нет отдельного брокера сообщений (RabbitMQ/Redis): очередь хранится в Postgres (`forecastrun`).
- Часть бизнес-логики рекомендаций rule-based и требует калибровки на production данных.
- Метрики качества модели (MAE/RMSE/WAPE) для online-мониторинга пока не вынесены в отдельный контур наблюдаемости.
