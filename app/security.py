"""CSRF protection, password policy, safe redirects, rate limiting, security headers."""
from __future__ import annotations

import re
import secrets
import time
from collections import defaultdict, deque
from functools import wraps
from urllib.parse import urljoin, urlparse

from flask import abort, current_app, flash, g, redirect, request, session, url_for

from .db import get_db

COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890", "qwerty123",
    "qwertyuiop", "iloveyou", "admin123", "welcome1", "letmein123", "abc12345", "11111111",
    "harvestguard", "harvest123", "farmer123", "plantvillage",
}
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


# --------------------------------------------------------------------------- CSRF
def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def _check_csrf():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        expected = session.get("_csrf", "")
        if not expected or not secrets.compare_digest(sent, expected):
            abort(400, description="Invalid or missing CSRF token. Reload the page and try again.")


# --------------------------------------------------------------------------- validation
def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email)) and len(email) <= 254


def password_problems(password: str, email: str = "", name: str = "") -> list[str]:
    issues = []
    if len(password) < 8:
        issues.append("at least 8 characters")
    if len(password) > 128:
        issues.append("at most 128 characters")
    if not re.search(r"[A-Za-z]", password):
        issues.append("at least one letter")
    if not re.search(r"\d", password):
        issues.append("at least one number")
    low = password.lower()
    if low in COMMON_PASSWORDS:
        issues.append("not a commonly used password")
    local = email.split("@")[0].lower() if email else ""
    if local and len(local) >= 4 and local in low:
        issues.append("must not contain your email name")
    return issues


def is_safe_next(target: str | None) -> bool:
    if not target:
        return False
    ref, test = urlparse(request.host_url), urlparse(urljoin(request.host_url, target))
    return test.scheme in ("http", "https") and ref.netloc == test.netloc and target.startswith("/")


# --------------------------------------------------------------------------- rate limiting
_hits: dict[str, deque] = defaultdict(deque)


def rate_limited(bucket: str, limit: int, window_s: int) -> bool:
    """In-memory sliding window per (bucket, client ip). Returns True if the request must be refused."""
    if current_app.config.get("TESTING_NO_RATELIMIT"):
        return False
    key = f"{bucket}:{request.remote_addr}"
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


# --------------------------------------------------------------------------- auth helpers
def current_user():
    return g.get("user")


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if g.get("user") is None:
            flash("Please sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return view(*a, **kw)
    return wrapped


def _load_user():
    uid = session.get("uid")
    g.user = None
    if uid is not None:
        row = get_db().execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (uid,)).fetchone()
        if row is None:
            session.clear()
        else:
            g.user = row


def init_app(app):
    app.before_request(_load_user)
    app.before_request(_check_csrf)
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'self'")
        if request.endpoint and request.endpoint not in ("static",):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp
