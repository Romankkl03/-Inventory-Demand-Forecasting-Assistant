from app import models  # noqa: F401  Ensures model metadata is registered.
from app.database.database import create_all_tables
from app.database.database import engine
from app.models import ModelType, ModelVersion, User, UserRole
from sqlmodel import Session, select


def init_db() -> None:
    create_all_tables()
    _seed_reference_data()


def _seed_reference_data() -> None:
    with Session(engine) as session:
        if session.exec(select(User)).first() is None:
            session.add(
                User(
                    name="System Admin",
                    email="admin@example.com",
                    role=UserRole.ADMIN,
                )
            )

        if session.exec(select(ModelVersion)).first() is None:
            session.add(
                ModelVersion(
                    name="baseline-demand-model",
                    version="1.0.0",
                    model_type=ModelType.BASELINE,
                    features_version="baseline-v1",
                    metrics_json={"mae": 0.0, "note": "bootstrap model"},
                )
            )
        session.commit()


if __name__ == "__main__":
    init_db()
