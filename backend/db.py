import hashlib
import secrets
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Iterator

import bcrypt
import psycopg
from psycopg.rows import dict_row
from settings import settings


def utcnow() -> datetime:
    return datetime.now(UTC)


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    connection = psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row)
    try:
        yield connection
    finally:
        connection.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session_token() -> tuple[str, str, datetime]:
    token = secrets.token_urlsafe(48)
    expires_at = utcnow() + timedelta(days=settings.session_ttl_days)
    return token, hash_token(token), expires_at


def ensure_schema() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT,
                    password_hash TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    bootstrap_admin_user()


def bootstrap_admin_user() -> None:
    username = settings.bootstrap_admin_username
    password = settings.bootstrap_admin_password

    if not username or not password:
        return

    first_name = settings.bootstrap_admin_first_name
    last_name = settings.bootstrap_admin_last_name
    email = settings.bootstrap_admin_email

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM users")
            user_count = cursor.fetchone()["count"]
            if user_count:
                return

            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, first_name, last_name, role)
                VALUES (%s, %s, %s, %s, %s, 'admin')
                """,
                (
                    username,
                    email,
                    hash_password(password),
                    first_name,
                    last_name,
                ),
            )


def get_user_by_username(username: str) -> dict | None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, email, password_hash, first_name, last_name, role, is_active, created_at
                FROM users
                WHERE username = %s
                """,
                (username,),
            )
            return cursor.fetchone()


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, email, first_name, last_name, role, is_active, created_at
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            return cursor.fetchone()


def list_users() -> list[dict]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, email, first_name, last_name, role, is_active, created_at
                FROM users
                ORDER BY created_at ASC, id ASC
                """
            )
            return list(cursor.fetchall())


def create_user(
    *,
    username: str,
    password: str,
    first_name: str,
    last_name: str,
    email: str | None,
    role: str,
) -> dict:
    password_hash = hash_password(password)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, first_name, last_name, role)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, username, email, first_name, last_name, role, is_active, created_at
                """,
                (username, email, password_hash, first_name, last_name, role),
            )
            return cursor.fetchone()


def create_session(user_id: int) -> tuple[str, datetime]:
    token, token_hash, expires_at = create_session_token()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO auth_sessions (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, token_hash, expires_at),
            )

    return token, expires_at


def get_user_by_token(token: str) -> dict | None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT u.id, u.username, u.email, u.first_name, u.last_name, u.role, u.is_active, u.created_at
                FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = %s
                  AND s.expires_at > NOW()
                """,
                (hash_token(token),),
            )
            return cursor.fetchone()


def delete_session(token: str) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM auth_sessions WHERE token_hash = %s",
                (hash_token(token),),
            )
