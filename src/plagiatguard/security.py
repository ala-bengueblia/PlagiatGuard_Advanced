from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from .config import JWT_SECRET, JWT_EXPIRE_MINUTES
from .storage import connect


password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError(
            "Le mot de passe doit contenir au moins 12 caractères."
        )

    return password_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return password_context.verify(password, hashed)


def create_user(username, password, role="analyst"):
    if role not in {"admin", "analyst", "viewer"}:
        raise ValueError("Rôle invalide.")

    hashed = hash_password(password)

    with connect() as conn:
        conn.execute(
            """INSERT INTO users
            (username, password_hash, role, active)
            VALUES (?, ?, ?, 1)""",
            (username, hashed, role),
        )


def authenticate(username, password):
    with connect() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND active=1",
            (username,),
        ).fetchone()

    if not user or not verify_password(
        password, user["password_hash"]
    ):
        return None

    now = datetime.now(timezone.utc)

    return jwt.encode(
        {
            "sub": username,
            "role": user["role"],
            "iat": now,
            "exp": now + timedelta(
                minutes=JWT_EXPIRE_MINUTES
            ),
        },
        JWT_SECRET,
        algorithm="HS256",
    )


def decode_token(token):
    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=["HS256"],
    )