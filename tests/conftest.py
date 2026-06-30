from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, select

import app.models  # noqa: F401
from app.api import app
from app.database.database import engine
from app.database.init_db import init_db
from app.models import Dataset, DatasetStatus, ModelVersion, SalesRecord, Store, User


@pytest.fixture(autouse=True)
def reset_db() -> None:
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    init_db()


@pytest.fixture
def session() -> Session:
    with Session(engine) as db_session:
        yield db_session


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded_forecast_input(session: Session) -> dict[str, int]:
    user = session.exec(select(User).where(User.email == "admin")).first()
    model = session.exec(select(ModelVersion).order_by(ModelVersion.id)).first()
    assert user is not None
    assert model is not None

    store = Store(external_id="seed-store-1")
    session.add(store)
    session.flush()

    session.add(
        SalesRecord(
            store_id=store.id,
            date=date(2026, 1, 1),
            sales=150.0,
            customers=12,
            promo=False,
            promo2=False,
            school_holiday=False,
            state_holiday="0",
            open=True,
        )
    )

    dataset = Dataset(
        name="seed-dataset",
        source="tests",
        uploaded_by=user.id,
        status=DatasetStatus.READY,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    return {
        "user_id": user.id,
        "dataset_id": dataset.id,
        "model_version_id": model.id,
        "store_id": store.id,
    }
