from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from app.database.database import get_session
from app.forecasting.data.reader import DataReader
from app.models import Forecast, ForecastRun, ModelVersion, SalesRecord, Store
from app.schemas import DatasetUploadRequest, ForecastRunRequest, RandomValForecastRequest, SalesRecordInput
from app.services import AuthService, DataService, ForecastingService, RecommendationService, ReportService

templates = Jinja2Templates(directory="app/view")
router = APIRouter(tags=["ui"])
SESSION_COOKIE_NAME = "session_token"


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(int(value)) if isinstance(value, (int, float)) else bool(value)


def _current_user(request: Request, session: Session):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return AuthService(session).resolve_user(token)


def _history_from_raw_rossmann(store_external_id: str, cutoff_date) -> tuple[list[str], list[float]]:
    try:
        raw_train = DataReader().read()["train"]
    except Exception:
        return [], []
    subset = raw_train[raw_train["Store"].astype(str) == str(store_external_id)].copy()
    if subset.empty:
        return [], []
    subset["Date"] = pd.to_datetime(subset["Date"]).dt.date
    subset = subset[subset["Date"] < cutoff_date].sort_values("Date").tail(30)
    return [d.isoformat() for d in subset["Date"]], [float(v) for v in subset["Sales"]]


def _to_ru_reason(tag: str) -> str:
    mapping = {
        "High recent demand": "повышенный недавний спрос",
        "Demand spike": "всплеск спроса",
        "Low inventory cover": "низкое покрытие запасом",
        "High forecast volatility": "высокая волатильность прогноза",
        "Demand growth vs previous period": "рост спроса к прошлому периоду",
        "Sufficient inventory": "достаточный запас",
    }
    return mapping.get(tag, tag.lower())


def _build_supplier_draft(user_name: str, run, recommendations):
    if run is None or not recommendations:
        return None
    total_expected = round(sum(item.expected_demand for item in recommendations), 2)
    total_order = round(sum(item.recommended_order for item in recommendations), 2)
    top_recommendation = recommendations[0]
    reasons = [_to_ru_reason(tag) for tag in top_recommendation.reason_tags] or ["плановое пополнение"]
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)
    subject = f"Заказ поставки на следующий период — магазин {top_recommendation.store_id}"
    body = (
        "Добрый день.\n"
        f"По результатам анализа спроса на следующий период для магазина {top_recommendation.store_id} "
        "рекомендуется пополнение поставки.\n"
        f"Период: {run.horizon} дней.\n"
        f"Ожидаемый спрос: {total_expected:,.0f} ед.\n"
        f"Рекомендуемый объем заказа: {total_order:,.0f} ед.\n"
        "Причины:\n"
        f"{reason_lines}\n"
        "Просим подтвердить возможность поставки на следующий период.\n\n"
        f"С уважением,\n{user_name}"
    )
    return {
        "subject": subject,
        "body": body,
    }


@router.get("/")
def ui_index(request: Request, session: Session = Depends(get_session)):
    user = _current_user(request, session)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "error": None},
    )


@router.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    try:
        token = AuthService(session).login(email=email, password=password)
    except Exception:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request, "error": "Invalid email or password."},
            status_code=401,
        )

    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(SESSION_COOKIE_NAME, token, httponly=True, samesite="lax")
    return response


@router.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={"request": request, "error": None},
    )


@router.post("/signup")
def signup_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    try:
        AuthService(session).signup(name=name, email=email, password=password)
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"request": request, "error": str(exc)},
            status_code=400,
        )
    return RedirectResponse("/login", status_code=302)


@router.get("/logout")
def logout(request: Request, session: Session = Depends(get_session)):
    AuthService(session).logout(request.cookies.get(SESSION_COOKIE_NAME))
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@router.get("/dashboard")
def dashboard(request: Request, session: Session = Depends(get_session)):
    user = _current_user(request, session)
    if user is None:
        return RedirectResponse("/login", status_code=302)

    run_id = request.query_params.get("run_id")
    selected_run = session.get(ForecastRun, int(run_id)) if run_id and run_id.isdigit() else None
    last_run = selected_run or session.exec(select(ForecastRun).order_by(ForecastRun.id.desc())).first()
    models = session.exec(select(ModelVersion).order_by(ModelVersion.id)).all()
    context = {
        "request": request,
        "user": user,
        "status_message": request.query_params.get("msg", ""),
        "error_message": request.query_params.get("err", ""),
        "last_run": last_run,
        "available_model_ids": [item.id for item in models] or [1],
        "history_forecast_chart": None,
        "metrics": None,
        "recommendations": None,
        "report": None,
    }

    if last_run is not None:
        forecast_rows = session.exec(
            select(Forecast)
            .where(Forecast.forecast_run_id == last_run.id)
            .order_by(Forecast.date)
        ).all()
        if forecast_rows:
            first_store_id = forecast_rows[0].store_id
            history_rows_db = session.exec(
                select(SalesRecord)
                .where(SalesRecord.store_id == first_store_id)
                .order_by(SalesRecord.date.desc())
            ).all()[:30]
            history_rows_db = list(reversed(history_rows_db))

            forecast_dates = [row.date.isoformat() for row in forecast_rows if row.store_id == first_store_id]
            forecast_values = [row.predicted_sales for row in forecast_rows if row.store_id == first_store_id]
            history_dates = [row.date.isoformat() for row in history_rows_db]
            history_values = [row.sales for row in history_rows_db]
            if not history_dates:
                store = session.get(Store, first_store_id)
                if store and forecast_dates:
                    history_dates, history_values = _history_from_raw_rossmann(
                        store.external_id, pd.to_datetime(forecast_dates[0]).date()
                    )

            context["history_forecast_chart"] = {
                "history_dates": history_dates,
                "history_values": history_values,
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values,
            }

            context["metrics"] = {
                "total_forecast_horizon": round(sum(forecast_values), 2),
                "avg_daily_forecast": round(sum(forecast_values) / max(1, len(forecast_values)), 2),
                "max_daily_forecast": round(max(forecast_values), 2),
            }

        recs = RecommendationService(session).get_or_create_recommendations(last_run.id)
        context["recommendations"] = recs.recommendations
        context["report"] = ReportService(session).get_or_create_report(last_run.id, created_by=user.id)
        context["supplier_draft"] = _build_supplier_draft(user.name, last_run, recs.recommendations)
    else:
        context["supplier_draft"] = None

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=context,
    )


@router.get("/dashboard/history")
def dashboard_history(request: Request, session: Session = Depends(get_session)):
    user = _current_user(request, session)
    if user is None:
        return RedirectResponse("/login", status_code=302)

    runs = session.exec(select(ForecastRun).order_by(ForecastRun.id.desc())).all()
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"request": request, "user": user, "runs": runs},
    )


@router.post("/dashboard/upload")
async def dashboard_upload(
    request: Request,
    dataset_name: str = Form(...),
    source: str = Form(default="uploaded-csv"),
    csv_file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    user = _current_user(request, session)
    if user is None:
        return RedirectResponse("/login", status_code=302)

    raw = await csv_file.read()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    records: list[SalesRecordInput] = []
    for _, row in df.iterrows():
        records.append(
            SalesRecordInput(
                store_external_id=str(int(row["Store"])),
                date=pd.to_datetime(row["Date"]).date(),
                sales=float(row["Sales"]),
                customers=int(row.get("Customers", 0)),
                promo=_as_bool(row.get("Promo", 0)),
                promo2=_as_bool(row.get("Promo2", 0)),
                school_holiday=_as_bool(row.get("SchoolHoliday", 0)),
                state_holiday=str(row.get("StateHoliday", "0")),
                open=_as_bool(row.get("Open", 1)),
            )
        )

    payload = DatasetUploadRequest(
        name=dataset_name,
        source=source,
        uploaded_by=user.id,
        records=records,
    )
    DataService(session).upload_dataset(payload)
    return RedirectResponse("/dashboard?msg=Датасет успешно загружен", status_code=302)


@router.post("/dashboard/random-val")
def dashboard_random_val(
    request: Request,
    model_version_id: int = Form(1),
    seed: int | None = Form(None),
    horizon: int = Form(14),
    session: Session = Depends(get_session),
):
    user = _current_user(request, session)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    ForecastingService(session).run_random_val_inference(
        RandomValForecastRequest(
            created_by=user.id,
            model_version_id=model_version_id,
            seed=seed,
            horizon=horizon,
        )
    )
    return RedirectResponse("/dashboard?msg=Прогноз на случайном примере val завершен", status_code=302)


@router.post("/dashboard/run-forecast")
def dashboard_run_forecast(
    request: Request,
    dataset_id: int = Form(...),
    model_version_id: int = Form(1),
    horizon: int = Form(14),
    session: Session = Depends(get_session),
):
    user = _current_user(request, session)
    if user is None:
        return RedirectResponse("/login", status_code=302)

    ForecastingService(session).run_forecast(
        ForecastRunRequest(
            dataset_id=dataset_id,
            model_version_id=model_version_id,
            created_by=user.id,
            horizon=horizon,
        )
    )
    return RedirectResponse("/dashboard?msg=Запуск прогноза завершен", status_code=302)
