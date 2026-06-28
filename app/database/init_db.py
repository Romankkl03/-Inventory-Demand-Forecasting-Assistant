from app import models  # noqa: F401  Ensures model metadata is registered.
from app.database.database import create_all_tables


def init_db() -> None:
    create_all_tables()


if __name__ == "__main__":
    init_db()
