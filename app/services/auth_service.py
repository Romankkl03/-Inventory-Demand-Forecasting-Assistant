"""Simple authentication service for UI login/signup flow."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models import User, UserCredential, UserRole, UserSession


class AuthService:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            120_000,
        )
        return digest.hex()

    def signup(self, *, name: str, email: str, password: str) -> User:
        if len(password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password should contain at least 6 characters.",
            )

        existing = self.session.exec(select(User).where(User.email == email)).first()
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists.",
            )

        user = User(name=name, email=email, role=UserRole.USER)
        self.session.add(user)
        self.session.flush()

        salt = secrets.token_hex(16)
        password_hash = f"{salt}${self._hash_password(password, salt)}"
        self.session.add(UserCredential(user_id=user.id, password_hash=password_hash))
        self.session.commit()
        self.session.refresh(user)
        return user

    def login(self, *, email: str, password: str) -> str:
        user = self.session.exec(select(User).where(User.email == email)).first()
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

        credential = self.session.exec(
            select(UserCredential).where(UserCredential.user_id == user.id)
        ).first()
        if credential is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

        try:
            salt, expected_hash = credential.password_hash.split("$", maxsplit=1)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Corrupted credential format.",
            ) from exc

        candidate_hash = self._hash_password(password, salt)
        if not secrets.compare_digest(candidate_hash, expected_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

        token = secrets.token_hex(24)
        self.session.add(UserSession(user_id=user.id, token=token))
        self.session.commit()
        return token

    def resolve_user(self, token: str | None) -> User | None:
        if not token:
            return None
        user_session = self.session.exec(
            select(UserSession).where(UserSession.token == token)
        ).first()
        if user_session is None:
            return None
        now = datetime.now(user_session.expires_at.tzinfo) if user_session.expires_at.tzinfo else datetime.utcnow()
        if user_session.expires_at < now:
            self.session.delete(user_session)
            self.session.commit()
            return None
        return self.session.get(User, user_session.user_id)

    def logout(self, token: str | None) -> None:
        if not token:
            return
        user_session = self.session.exec(
            select(UserSession).where(UserSession.token == token)
        ).first()
        if user_session:
            self.session.delete(user_session)
            self.session.commit()
