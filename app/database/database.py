from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.database.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.sqlalchemy_database_url,
    echo=settings.DB_ECHO,
    pool_pre_ping=True,
)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def create_all_tables() -> None:
    SQLModel.metadata.create_all(engine)
