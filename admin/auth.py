import hashlib
import hmac
import os
import secrets
import time
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Cookie, HTTPException, Request
from loguru import logger
from sqlalchemy import text

from db import get_engine

SESSION_COOKIE = "admin_session"
CSRF_HEADER = "x-csrf-token"
SESSION_TTL_HOURS = int(os.getenv("ADMIN_SESSION_TTL_HOURS", "12"))
LOCKOUT_THRESHOLD = int(os.getenv("ADMIN_LOCKOUT_THRESHOLD", "5"))
LOCKOUT_SECONDS = int(os.getenv("ADMIN_LOCKOUT_SECONDS", "300"))
_ph = PasswordHasher()

# In-memory fallback for tests and single process deploys without DB rows.
_memory_sessions: dict[str, dict] = {}
_login_failures: dict[str, list[float]] = {}


def hash_password(password: str) -> str:
    return _ph.hash_secret(secret=password.encode("utf-8")).hash


def verify_password(stored_hash: str, candidate: str) -> bool:
    try:
        ok = _ph.verify(stored_hash, candidate.encode("utf-8"))
    except VerifyMismatchError:
        ok = False
    except Exception:
        ok = False
    dummy = "argon2id$v=19$m=65536,t=3,p=4$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    try:
        if not stored_hash:
            _ph.verify(dummy, candidate.encode("utf-8"))
            return False
    except Exception:
        pass
    return bool(ok)


def _is_locked_out(email: str) -> bool:
    now = time.time()
    fails = [t for t in _login_failures.get(email, []) if now - t < LOCKOUT_SECONDS]
    _login_failures[email] = fails
    return len(fails) >= LOCKOUT_THRESHOLD


def _record_failure(email: str) -> None:
    _login_failures.setdefault(email, []).append(time.time())


def _record_success(email: str) -> None:
    _login_failures[email] = []


def generate_session():
    return secrets.token_urlsafe(32), secrets.token_urlsafe(32)


def get_current_admin(request: Request, admin_session: str | None = Cookie(default=None)):
    sid = admin_session or request.headers.get("x-admin-session")
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    sess = _lookup_session(sid)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sess


def _lookup_session(sid: str):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("""SELECT s.admin_id, s.csrf_token, s.expires_at, u.email, u.role
                FROM admin_sessions s JOIN admin_users u ON u.id = s.admin_id
                WHERE s.session_id = :sid"""),
                {"sid": sid},
            ).mappings().first()
            if not row:
                raise HTTPException(status_code=401, detail="Not authenticated")
            exp = row["expires_at"]
            if exp and exp.tzinfo is not None and exp < datetime.now(UTC):
                raise HTTPException(status_code=401, detail="Session expired")
            if exp and exp.tzinfo is None and exp < datetime.now(UTC).replace(tzinfo=None):
                raise HTTPException(status_code=401, detail="Session expired")
            return {"admin_id": row["admin_id"], "csrf_token": row["csrf_token"], "email": row["email"], "role": row["role"]}
    except HTTPException:
        raise
    except Exception:
        sess = _memory_sessions.get(sid)
        if not sess or sess["expires_at"] < datetime.now(UTC):
            raise HTTPException(status_code=401, detail="Not authenticated") from None
        return sess


def require_csrf(request: Request, session: dict) -> None:
    if request.method not in ("POST", "PATCH", "PUT", "DELETE"):
        return
    token = request.headers.get(CSRF_HEADER) or request.headers.get("X-CSRF-Token")
    expected = (session or {}).get("csrf_token", "")
    if not token or not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def login(email: str, password: str) -> tuple[str, str]:
    email_norm = (email or "").strip().lower()
    if _is_locked_out(email_norm):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")
    user = _find_user(email_norm)
    ok = verify_password(user["password_hash"] if user else "", password or "")
    if not ok or user is None:
        _record_failure(email_norm)
        raise HTTPException(status_code=401, detail="Invalid email or password")
    _record_success(email_norm)
    sid, csrf = generate_session()
    expires = datetime.now(UTC) + timedelta(hours=SESSION_TTL_HOURS)
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO admin_sessions (session_id, admin_id, csrf_token, expires_at) VALUES (:sid, :admin_id, :csrf, :exp)"),
                {"sid": sid, "admin_id": user["id"], "csrf": csrf, "exp": expires},
            )
            conn.execute(text("UPDATE admin_users SET last_login_at = now() WHERE id = :id"), {"id": user["id"]})
    except Exception as exc:
        logger.warning("session persist failed, using memory: {}", exc)
        _memory_sessions[sid] = {"admin_id": user["id"], "csrf_token": csrf, "email": user["email"], "role": user["role"], "expires_at": expires}
    return sid, csrf


def logout(sid: str | None) -> None:
    if not sid:
        return
    _memory_sessions.pop(sid, None)
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM admin_sessions WHERE session_id = :sid"), {"sid": sid})
    except Exception:
        pass


def _find_user(email: str):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(text("SELECT id, email, password_hash, role FROM admin_users WHERE email = :email"), {"email": email}).mappings().first()
            return dict(row) if row else None
    except Exception:
        return None


def create_admin(email: str, password: str) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm or "@" not in email_norm:
        raise ValueError("Valid email required")
    if not password or len(password) < 12:
        raise ValueError("Password must be at least 12 characters")
    h = hash_password(password)
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("INSERT INTO admin_users (email, password_hash, role) VALUES (:email, :h, :role) ON CONFLICT (email) DO UPDATE SET password_hash = :h RETURNING id"),
            {"email": email_norm, "h": h, "role": "admin"},
        ).mappings().first()
    return int(row["id"])


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
