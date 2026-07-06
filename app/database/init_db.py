from app import models  # noqa: F401  Ensures model metadata is registered.
from app.database.database import create_all_tables
from app.database.database import engine
from app.models import ModelType, ModelVersion, User, UserCredential, UserRole
import hashlib
from sqlmodel import Session, select


def init_db() -> None:
    create_all_tables()
    _seed_reference_data()


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return digest


def _ensure_bootstrap_admin(
    session: Session,
    *,
    email: str,
    password: str,
    name: str,
) -> None:
    admin = session.exec(select(User).where(User.email == email)).first()
    if admin is None:
        admin = User(name=name, email=email, role=UserRole.ADMIN)
        session.add(admin)
        session.flush()
    else:
        admin.role = UserRole.ADMIN
        if not admin.name:
            admin.name = name

    # Demo bootstrap account: keep credentials deterministic.
    salt = "bootstrap_admin_salt"
    password_hash = f"{salt}${_hash_password(password, salt)}"
    admin_cred = session.exec(
        select(UserCredential).where(UserCredential.user_id == admin.id)
    ).first()
    if admin_cred is None:
        session.add(
            UserCredential(
                user_id=admin.id,
                password_hash=password_hash,
            )
        )
    else:
        admin_cred.password_hash = password_hash


def _seed_reference_data() -> None:
    with Session(engine) as session:
        _ensure_bootstrap_admin(
            session,
            email="admin@admin",
            password="admin123",
            name="admin",
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
