from app import models  # noqa: F401  Ensures model metadata is registered.
from app.database.database import create_all_tables
from app.database.database import engine
from app.models import ModelType, ModelVersion, User, UserCredential, UserRole
import hashlib
from sqlmodel import Session, select


def init_db() -> None:
    create_all_tables()
    _seed_reference_data()


def _seed_reference_data() -> None:
    with Session(engine) as session:
        admin = session.exec(select(User).where(User.email == "admin")).first()
        if admin is None:
            admin = User(name="admin", email="admin", role=UserRole.ADMIN)
            session.add(admin)
            session.flush()

        admin_cred = session.exec(
            select(UserCredential).where(UserCredential.user_id == admin.id)
        ).first()
        if admin_cred is None:
            salt = "admin_salt"
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                b"admin",
                salt.encode("utf-8"),
                120_000,
            ).hex()
            session.add(
                UserCredential(
                    user_id=admin.id,
                    password_hash=f"{salt}${digest}",
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
