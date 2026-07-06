"""Authentication-related SQLModel entities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserCredential(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", unique=True, index=True)
    password_hash: str = Field(min_length=32, max_length=256)
    created_at: datetime = Field(default_factory=utc_now)


class UserSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    token: str = Field(unique=True, index=True, min_length=32, max_length=128)
    expires_at: datetime = Field(default_factory=lambda: utc_now() + timedelta(days=7))
    created_at: datetime = Field(default_factory=utc_now)
