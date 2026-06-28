"""SQLModel domain entities for the ID forecasting assistant.

Defines database tables, enums, and relationships for users, stores,
sales data, forecasting workflows, and subscription billing.
"""

from datetime import date as DateType
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel


def utc_now() -> datetime:
    """Return the current UTC datetime with timezone information.

    Returns:
        datetime: Current moment as a timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


class UserRole(str, Enum):
    """Application roles that define user permissions.

    Values:
        USER: Standard user with basic access.
        ANALYST: User who can run forecasts and view reports.
        ADMIN: User with full administrative access.
    """

    USER = "user"
    ANALYST = "analyst"
    ADMIN = "admin"


class DatasetStatus(str, Enum):
    """Lifecycle states of an uploaded dataset.

    Values:
        NEW: Dataset was uploaded and awaits validation.
        VALIDATING: Dataset is being checked for schema and quality.
        READY: Dataset passed validation and can be used in forecast runs.
        FAILED: Dataset validation failed.
    """

    NEW = "new"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"


class ForecastRunStatus(str, Enum):
    """Execution states of a forecast run job.

    Values:
        QUEUED: Run is waiting to be picked up by a worker.
        RUNNING: Run is currently in progress.
        COMPLETED: Run finished successfully.
        FAILED: Run terminated with an error.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, Enum):
    """Supported forecasting model families.

    Values:
        BASELINE: Simple statistical or rule-based baseline model.
        GRADIENT_BOOSTING: Gradient boosting regressor.
        RANDOM_FOREST: Random forest regressor.
        CUSTOM: User-defined or externally registered model.
    """

    BASELINE = "baseline"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk classification for inventory recommendations.

    Values:
        LOW: Low risk of stockout or overstock.
        MEDIUM: Moderate risk requiring attention.
        HIGH: High risk requiring immediate action.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SubscriptionStatus(str, Enum):
    """Lifecycle states of a user subscription.

    Values:
        ACTIVE: Subscription is currently valid and billable.
        TRIAL: User is in a trial period.
        PAUSED: Billing is paused; access may be limited.
        CANCELED: Subscription was canceled and will not renew.
        EXPIRED: Subscription period has ended.
    """

    ACTIVE = "active"
    TRIAL = "trial"
    PAUSED = "paused"
    CANCELED = "canceled"
    EXPIRED = "expired"


class PaymentStatus(str, Enum):
    """Processing states of a payment transaction.

    Values:
        PENDING: Payment was initiated but not yet confirmed.
        SUCCEEDED: Payment completed successfully.
        FAILED: Payment was rejected or failed.
        REFUNDED: Payment was refunded to the customer.
    """

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"


class User(SQLModel, table=True):
    """Platform user account.

    Attributes:
        id: Primary key.
        name: Display name of the user.
        email: Unique login email address.
        role: Permission role assigned to the user.
        created_at: Timestamp when the account was created.
        datasets: Datasets uploaded by this user.
        forecast_runs: Forecast runs initiated by this user.
        reports: Reports created by this user.
        subscriptions: Active and historical subscriptions.
        payments: Payment transactions linked to this user.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(min_length=1, max_length=255)
    email: str = Field(unique=True, index=True, min_length=5, max_length=255)
    role: UserRole = Field(default=UserRole.USER)
    created_at: datetime = Field(default_factory=utc_now)

    datasets: list["Dataset"] = Relationship(back_populates="uploader")
    forecast_runs: list["ForecastRun"] = Relationship(back_populates="creator")
    reports: list["Report"] = Relationship(back_populates="creator")
    subscriptions: list["Subscription"] = Relationship(back_populates="user")
    payments: list["Payment"] = Relationship(back_populates="user")


class Store(SQLModel, table=True):
    """Retail store with metadata used for forecasting.

    Attributes:
        id: Primary key.
        external_id: External identifier from the source dataset (e.g. Rossmann store id).
        store_type: Store format or category code.
        assortment: Assortment level code.
        competition_distance: Distance in meters to the nearest competitor.
        sales_records: Historical daily sales rows for this store.
        forecasts: Point forecasts generated for this store.
        recommendations: Inventory recommendations for this store.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: str = Field(index=True, unique=True, min_length=1, max_length=64)
    store_type: Optional[str] = Field(default=None, max_length=32)
    assortment: Optional[str] = Field(default=None, max_length=32)
    competition_distance: Optional[float] = Field(default=None, ge=0)

    sales_records: list["SalesRecord"] = Relationship(back_populates="store")
    forecasts: list["Forecast"] = Relationship(back_populates="store")
    recommendations: list["Recommendation"] = Relationship(back_populates="store")


class SalesRecord(SQLModel, table=True):
    """Daily sales observation for a single store.

    Attributes:
        id: Primary key.
        store_id: Foreign key to the store.
        date: Calendar date of the observation.
        sales: Total sales amount for the day.
        customers: Number of customers served.
        promo: Whether a promotion was active.
        promo2: Whether a secondary promotion was active.
        school_holiday: Whether a school holiday occurred.
        state_holiday: State holiday code (``"0"`` if none).
        open: Whether the store was open for business.
        store: Related store entity.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    store_id: int = Field(foreign_key="store.id", index=True)
    date: DateType = Field(index=True)
    sales: float = Field(ge=0)
    customers: int = Field(ge=0)
    promo: bool = Field(default=False)
    promo2: bool = Field(default=False)
    school_holiday: bool = Field(default=False)
    state_holiday: str = Field(default="0", max_length=16)
    open: bool = Field(default=True)

    store: Store = Relationship(back_populates="sales_records")


class Dataset(SQLModel, table=True):
    """Uploaded dataset available for forecast runs.

    Attributes:
        id: Primary key.
        name: Human-readable dataset name.
        source: URI or path to the raw data file.
        uploaded_by: Foreign key to the uploading user.
        status: Current validation lifecycle state.
        created_at: Timestamp when the dataset was uploaded.
        uploader: User who uploaded the dataset.
        forecast_runs: Forecast runs that used this dataset.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(min_length=1, max_length=255)
    source: str = Field(min_length=1, max_length=512)
    uploaded_by: int = Field(foreign_key="user.id", index=True)
    status: DatasetStatus = Field(default=DatasetStatus.NEW)
    created_at: datetime = Field(default_factory=utc_now)

    uploader: User = Relationship(back_populates="datasets")
    forecast_runs: list["ForecastRun"] = Relationship(back_populates="dataset")


class ModelVersion(SQLModel, table=True):
    """Registered version of a trained forecasting model.

    Attributes:
        id: Primary key.
        name: Model name (e.g. ``"xgboost_v1"``).
        version: Semantic or build version string.
        model_type: Family of the underlying algorithm.
        features_version: Identifier of the feature pipeline used at training time.
        metrics_json: Validation metrics stored as a JSON object.
        created_at: Timestamp when the model version was registered.
        forecast_runs: Forecast runs that used this model version.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=64)
    model_type: ModelType = Field(default=ModelType.BASELINE)
    features_version: str = Field(min_length=1, max_length=64)
    metrics_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )
    created_at: datetime = Field(default_factory=utc_now)

    forecast_runs: list["ForecastRun"] = Relationship(back_populates="model_version")


class ForecastRun(SQLModel, table=True):
    """End-to-end forecasting job for a dataset and model version.

    Attributes:
        id: Primary key.
        dataset_id: Foreign key to the input dataset.
        model_version_id: Foreign key to the model used for prediction.
        created_by: Foreign key to the user who started the run.
        status: Current execution state.
        horizon: Number of days or weeks in the forecast horizon.
        started_at: Timestamp when processing began (``None`` if not started).
        finished_at: Timestamp when processing ended (``None`` if not finished).
        dataset: Input dataset for this run.
        model_version: Model version used for this run.
        creator: User who initiated this run.
        forecasts: Generated point forecasts.
        recommendations: Inventory recommendations derived from forecasts.
        reports: Summary reports produced for this run.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    model_version_id: int = Field(foreign_key="modelversion.id", index=True)
    created_by: int = Field(foreign_key="user.id", index=True)
    status: ForecastRunStatus = Field(default=ForecastRunStatus.QUEUED)
    horizon: int = Field(ge=1, description="Number of days/weeks in forecast horizon")
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)

    dataset: Dataset = Relationship(back_populates="forecast_runs")
    model_version: ModelVersion = Relationship(back_populates="forecast_runs")
    creator: User = Relationship(back_populates="forecast_runs")
    forecasts: list["Forecast"] = Relationship(back_populates="forecast_run")
    recommendations: list["Recommendation"] = Relationship(back_populates="forecast_run")
    reports: list["Report"] = Relationship(back_populates="forecast_run")


class Forecast(SQLModel, table=True):
    """Point forecast for a store on a specific date.

    Attributes:
        id: Primary key.
        forecast_run_id: Foreign key to the parent forecast run.
        store_id: Foreign key to the target store.
        date: Date for which sales are predicted.
        predicted_sales: Predicted sales amount (non-negative).
        forecast_run: Parent forecast run.
        store: Target store.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    forecast_run_id: int = Field(foreign_key="forecastrun.id", index=True)
    store_id: int = Field(foreign_key="store.id", index=True)
    date: DateType = Field(index=True)
    predicted_sales: float = Field(ge=0)

    forecast_run: ForecastRun = Relationship(back_populates="forecasts")
    store: Store = Relationship(back_populates="forecasts")


class Recommendation(SQLModel, table=True):
    """Inventory order recommendation derived from a forecast run.

    Attributes:
        id: Primary key.
        forecast_run_id: Foreign key to the source forecast run.
        store_id: Foreign key to the target store.
        expected_demand: Forecasted demand used for the recommendation.
        recommended_order: Suggested order quantity.
        risk_level: Risk classification of stockout or overstock.
        comment: Optional analyst note or explanation.
        created_at: Timestamp when the recommendation was generated.
        forecast_run: Source forecast run.
        store: Target store.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    forecast_run_id: int = Field(foreign_key="forecastrun.id", index=True)
    store_id: int = Field(foreign_key="store.id", index=True)
    expected_demand: float = Field(ge=0)
    recommended_order: float = Field(ge=0)
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    comment: Optional[str] = Field(default=None, max_length=1000)
    created_at: datetime = Field(default_factory=utc_now)

    forecast_run: ForecastRun = Relationship(back_populates="recommendations")
    store: Store = Relationship(back_populates="recommendations")


class Report(SQLModel, table=True):
    """Summary report for a completed forecast run.

    Attributes:
        id: Primary key.
        forecast_run_id: Foreign key to the related forecast run.
        created_by: Foreign key to the user who created the report.
        summary: Text summary of results and key findings.
        created_at: Timestamp when the report was created.
        forecast_run: Related forecast run.
        creator: User who authored the report.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    forecast_run_id: int = Field(foreign_key="forecastrun.id", index=True)
    created_by: int = Field(foreign_key="user.id", index=True)
    summary: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    forecast_run: ForecastRun = Relationship(back_populates="reports")
    creator: User = Relationship(back_populates="reports")


class TariffPlan(SQLModel, table=True):
    """Subscription pricing plan with associated limits and features.

    Attributes:
        id: Primary key.
        name: Unique plan name shown to users.
        price_month: Monthly price in the default currency.
        description: Marketing or product description.
        is_active: Whether the plan is available for new subscriptions.
        created_at: Timestamp when the plan was created.
        subscriptions: Active subscriptions on this plan.
        usage_limits: Resource quotas tied to this plan.
        features: Feature flags enabled for this plan.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(min_length=1, max_length=255, unique=True, index=True)
    price_month: float = Field(ge=0)
    description: str = Field(min_length=1)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utc_now)

    subscriptions: list["Subscription"] = Relationship(back_populates="tariff_plan")
    usage_limits: list["UsageLimit"] = Relationship(back_populates="tariff_plan")
    features: list["SubscriptionFeature"] = Relationship(back_populates="tariff_plan")


class Subscription(SQLModel, table=True):
    """User subscription to a tariff plan.

    Attributes:
        id: Primary key.
        user_id: Foreign key to the subscribing user.
        tariff_plan_id: Foreign key to the selected plan.
        status: Current subscription lifecycle state.
        start_date: First day the subscription is valid.
        end_date: Last valid day (``None`` for open-ended subscriptions).
        auto_renew: Whether the subscription renews automatically.
        created_at: Timestamp when the subscription was created.
        user: Subscribing user.
        tariff_plan: Associated pricing plan.
        payments: Payment history for this subscription.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    tariff_plan_id: int = Field(foreign_key="tariffplan.id", index=True)
    status: SubscriptionStatus = Field(default=SubscriptionStatus.ACTIVE)
    start_date: DateType
    end_date: Optional[DateType] = Field(default=None)
    auto_renew: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utc_now)

    user: User = Relationship(back_populates="subscriptions")
    tariff_plan: TariffPlan = Relationship(back_populates="subscriptions")
    payments: list["Payment"] = Relationship(back_populates="subscription")


class Payment(SQLModel, table=True):
    """Payment transaction for a subscription.

    Attributes:
        id: Primary key.
        subscription_id: Foreign key to the billed subscription.
        user_id: Foreign key to the paying user.
        amount: Charged amount (non-negative).
        currency: ISO currency code (default ``"RUB"``).
        status: Current payment processing state.
        payment_date: Timestamp when the payment was recorded.
        provider: Payment gateway or processor name.
        transaction_id: Unique identifier from the payment provider.
        subscription: Billed subscription.
        user: Paying user.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    subscription_id: int = Field(foreign_key="subscription.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    amount: float = Field(ge=0)
    currency: str = Field(default="RUB", min_length=3, max_length=8)
    status: PaymentStatus = Field(default=PaymentStatus.PENDING)
    payment_date: datetime = Field(default_factory=utc_now)
    provider: str = Field(min_length=1, max_length=128)
    transaction_id: str = Field(min_length=1, max_length=255, unique=True, index=True)

    subscription: Subscription = Relationship(back_populates="payments")
    user: User = Relationship(back_populates="payments")


class UsageLimit(SQLModel, table=True):
    """Monthly resource quotas for a tariff plan.

    Attributes:
        id: Primary key.
        tariff_plan_id: Foreign key to the parent plan.
        max_forecast_runs_per_month: Maximum forecast runs allowed per month.
        max_stores: Maximum number of stores that can be forecasted.
        max_reports: Maximum number of reports that can be generated.
        created_at: Timestamp when the limits were defined.
        tariff_plan: Parent tariff plan.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    tariff_plan_id: int = Field(foreign_key="tariffplan.id", index=True)
    max_forecast_runs_per_month: int = Field(ge=0)
    max_stores: int = Field(ge=0)
    max_reports: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    tariff_plan: TariffPlan = Relationship(back_populates="usage_limits")


class SubscriptionFeature(SQLModel, table=True):
    """Feature flag included in a tariff plan.

    Attributes:
        id: Primary key.
        tariff_plan_id: Foreign key to the parent plan.
        feature_name: Identifier of the enabled or disabled feature.
        is_enabled: Whether the feature is active for subscribers.
        tariff_plan: Parent tariff plan.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    tariff_plan_id: int = Field(foreign_key="tariffplan.id", index=True)
    feature_name: str = Field(min_length=1, max_length=128)
    is_enabled: bool = Field(default=True)

    tariff_plan: TariffPlan = Relationship(back_populates="features")
