import base64
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from utils.json_storage import DATA_DIR, StorageError, read_json, update_json


USERS_FILE = DATA_DIR / "users.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"
USER_DEFAULT = {"users": []}
SESSIONS_DEFAULT = {"sessions": []}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SESSION_TTL_DAYS = 7


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return (value or "").strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    )
    return "scrypt$16384$8$1${}${}".format(
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(key).decode("ascii"),
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, n, r, p, salt_b64, key_b64 = encoded_hash.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(key_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def public_user(user):
    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
        "username": user["username"],
        "created_at": user.get("created_at"),
    }


def validate_registration(full_name, email, username, password, confirm_password):
    errors = []
    if len((full_name or "").strip()) < 2:
        errors.append("Full name must contain at least 2 characters.")
    if not EMAIL_RE.match(email or ""):
        errors.append("Enter a valid email address.")
    if len((username or "").strip()) < 3:
        errors.append("Username must contain at least 3 characters.")
    if len(password or "") < 8:
        errors.append("Password must be at least 8 characters.")
    if password and not re.search(r"[A-Za-z]", password):
        errors.append("Password must include at least one letter.")
    if password and not re.search(r"\d", password):
        errors.append("Password must include at least one number.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    return errors


def register_user(full_name, email, username, password, confirm_password):
    email_norm = normalize(email)
    username_norm = normalize(username)
    errors = validate_registration(full_name, email_norm, username_norm, password, confirm_password)
    if errors:
        return False, errors, None

    def mutator(data):
        users = data.setdefault("users", [])
        if any(normalize(user.get("email")) == email_norm for user in users):
            return False, ["An account with this email already exists."], None
        if any(normalize(user.get("username")) == username_norm for user in users):
            return False, ["This username is already taken."], None
        user = {
            "id": str(uuid4()),
            "full_name": full_name.strip(),
            "email": email_norm,
            "username": username_norm,
            "password_hash": hash_password(password),
            "created_at": utc_now(),
            "last_login_at": None,
        }
        users.append(user)
        return True, [], public_user(user)

    return update_json(USERS_FILE, USER_DEFAULT, mutator)


def authenticate(identifier, password):
    identifier_norm = normalize(identifier)
    try:
        data = read_json(USERS_FILE, USER_DEFAULT)
    except StorageError as error:
        return False, str(error), None

    for user in data.get("users", []):
        matches = normalize(user.get("email")) == identifier_norm or normalize(user.get("username")) == identifier_norm
        if matches and verify_password(password or "", user.get("password_hash", "")):
            user_id = user["id"]

            def mark_login(payload):
                for item in payload.get("users", []):
                    if item.get("id") == user_id:
                        item["last_login_at"] = utc_now()
                        return public_user(item)
                return public_user(user)

            return True, "", update_json(USERS_FILE, USER_DEFAULT, mark_login)
    return False, "Invalid email, username, or password.", None


def get_user_by_id(user_id):
    data = read_json(USERS_FILE, USER_DEFAULT)
    for user in data.get("users", []):
        if user.get("id") == user_id:
            return public_user(user)
    return None


def _session_is_active(session, now):
    try:
        return datetime.fromisoformat(session["expires_at"]) > now
    except (KeyError, TypeError, ValueError):
        return False


def create_session(user_id):
    """Create a long-lived, revocable browser session for an authenticated user."""
    now = datetime.now(timezone.utc)
    token = secrets.token_urlsafe(32)
    record = {
        "token": token,
        "user_id": user_id,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=SESSION_TTL_DAYS)).isoformat(),
    }

    def mutator(data):
        sessions = data.setdefault("sessions", [])
        sessions[:] = [session for session in sessions if _session_is_active(session, now)]
        sessions.append(record)
        return token

    return update_json(SESSIONS_FILE, SESSIONS_DEFAULT, mutator)


def get_session_user(token):
    """Return the public user for a valid session token, otherwise ``None``."""
    if not isinstance(token, str) or not token:
        return None
    now = datetime.now(timezone.utc)
    data = read_json(SESSIONS_FILE, SESSIONS_DEFAULT)
    for session in data.get("sessions", []):
        if hmac.compare_digest(str(session.get("token", "")), token) and _session_is_active(session, now):
            return get_user_by_id(session.get("user_id"))
    return None


def revoke_session(token):
    if not isinstance(token, str) or not token:
        return False

    def mutator(data):
        sessions = data.setdefault("sessions", [])
        before = len(sessions)
        sessions[:] = [session for session in sessions if not hmac.compare_digest(str(session.get("token", "")), token)]
        return len(sessions) != before

    return update_json(SESSIONS_FILE, SESSIONS_DEFAULT, mutator)
