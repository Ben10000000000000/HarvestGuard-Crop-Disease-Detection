"""Sign-up, login, logout, password reset and account management."""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import shutil
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

from flask import (Blueprint, current_app, flash, g, redirect, render_template, request, session,
                   url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from .db import audit, get_db
from .security import (is_safe_next, login_required, password_problems, rate_limited, valid_email)

bp = Blueprint("auth", __name__)
log = logging.getLogger("harvestguard.auth")
UTC_FMT = "%Y-%m-%d %H:%M:%S"
_DUMMY_HASH = generate_password_hash("dummy-password-for-timing")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fmt(dt: datetime) -> str:
    return dt.strftime(UTC_FMT)


def _start_session(user_id: int, remember: bool):
    session.clear()                 # prevents session fixation
    session["uid"] = user_id
    session.permanent = remember


# --------------------------------------------------------------------------- OTP helpers
def _is_verified(user) -> bool:
    """Accounts created before OTP existed have no 'verified' field - treat them as verified."""
    try:
        return bool(user["verified"])
    except (KeyError, IndexError):
        return True


def _hash_otp(user_id: int, purpose: str, code: str) -> str:
    key = current_app.config["SECRET_KEY"].encode()
    return hmac.new(key, f"{user_id}:{purpose}:{code}".encode(), hashlib.sha256).hexdigest()


def _send_mail(to: str, subject: str, body: str) -> bool:
    host = os.environ.get("HG_SMTP_HOST")
    if not host:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("HG_SMTP_FROM", os.environ.get("HG_SMTP_USER", "no-reply@harvestguard.local"))
    msg["To"] = to
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.environ.get("HG_SMTP_PORT", "587")), timeout=15) as s:
            s.starttls()
            if os.environ.get("HG_SMTP_USER"):
                s.login(os.environ["HG_SMTP_USER"], os.environ.get("HG_SMTP_PASSWORD", ""))
            s.send_message(msg)
        return True
    except Exception as exc:  # network / auth problems must never break the page
        log.error("SMTP send failed: %s", exc)
        return False


_PURPOSE_TEXT = {
    "verify": ("Verify your HarvestGuard email", "Use this code to verify your email address"),
    "reset": ("Your HarvestGuard password reset code", "Use this code to reset your password"),
}


def _issue_otp(user_id: int, name: str, email: str, purpose: str) -> str | None:
    """Create a fresh 6-digit code (invalidating older ones) and e-mail it.

    Returns an error message if the resend cool-down is still running, else None.
    """
    db = get_db()
    cfg = current_app.config
    last = db.execute("SELECT * FROM otps WHERE user_id = ? AND purpose = ? AND used = 0 ORDER BY id DESC LIMIT 1",
                      (user_id, purpose)).fetchone()
    if last and cfg["OTP_RESEND_SECONDS"] > 0:
        wait = cfg["OTP_RESEND_SECONDS"] - (_now() - datetime.strptime(last["created_at"], UTC_FMT)).total_seconds()
        if wait > 0:
            return f"Please wait {int(wait) + 1} seconds before asking for another code."
    code = f"{secrets.randbelow(10 ** 6):06d}"
    db.execute("UPDATE otps SET used = 1 WHERE user_id = ? AND purpose = ?", (user_id, purpose))
    db.execute("INSERT INTO otps (user_id, purpose, code_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
               (user_id, purpose, _hash_otp(user_id, purpose, code),
                _fmt(_now() + timedelta(minutes=cfg["OTP_MINUTES"])), _fmt(_now())))
    db.commit()
    subject, lead = _PURPOSE_TEXT[purpose]
    body = (f"Hi {name.split()[0]},\n\n{lead}:\n\n    {code}\n\n"
            f"The code expires in {cfg['OTP_MINUTES']} minutes and works once. "
            "If you did not request it, you can ignore this email - nobody can use it without the code.")
    if not _send_mail(email, subject, body):
        log.warning("OTP (%s) for %s: %s", purpose, email, code)
        print(f"\n[HarvestGuard] {purpose} OTP for {email}: {code}\n", flush=True)
        if cfg["SHOW_OTP"]:
            session["_dev_otp"] = code
    return None


def _check_otp(user_id: int, purpose: str, code: str) -> str | None:
    """Return None if the code is right (and burn it), otherwise a user-facing error."""
    db = get_db()
    row = db.execute("SELECT * FROM otps WHERE user_id = ? AND purpose = ? AND used = 0 ORDER BY id DESC LIMIT 1",
                     (user_id, purpose)).fetchone()
    bad = "That code is incorrect or has expired. Check the latest email, or request a new code."
    if row is None or datetime.strptime(row["expires_at"], UTC_FMT) <= _now():
        return bad
    if row["attempts"] >= current_app.config["OTP_MAX_ATTEMPTS"]:
        db.execute("UPDATE otps SET used = 1 WHERE id = ?", (row["id"],)); db.commit()
        return "Too many wrong attempts. Please request a new code."
    if not hmac.compare_digest(row["code_hash"], _hash_otp(user_id, purpose, code)):
        db.execute("UPDATE otps SET attempts = attempts + 1 WHERE id = ?", (row["id"],)); db.commit()
        return bad
    db.execute("UPDATE otps SET used = 1 WHERE id = ?", (row["id"],)); db.commit()
    return None


def _render_otp(purpose: str, email: str, status: int = 200):
    return render_template("otp.html", purpose=purpose, email=email, dev_otp=session.get("_dev_otp")), status


# --------------------------------------------------------------------------- sign up
@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(url_for("main.dashboard"))
    form = {"name": "", "email": ""}
    if request.method == "POST":
        if rate_limited("signup", 10, 3600):
            flash("Too many sign-up attempts from this network. Please try again later.", "error")
            return render_template("signup.html", form=form), 429
        name = " ".join(request.form.get("name", "").split())
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        form = {"name": name, "email": email}
        errors = []
        if not (2 <= len(name) <= 60):
            errors.append("Enter your name (2-60 characters).")
        if not valid_email(email):
            errors.append("Enter a valid email address.")
        probs = password_problems(password, email, name)
        if probs:
            errors.append("Password needs: " + ", ".join(probs) + ".")
        if password != confirm:
            errors.append("The two passwords do not match.")
        db = get_db()
        existing = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not errors and existing and _is_verified(existing):
            errors.append("An account with this email already exists. Try signing in instead.")
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("signup.html", form=form), 400
        if existing:   # earlier sign-up never confirmed: the person now proving ownership takes it over
            db.execute("UPDATE users SET name = ?, password_hash = ? WHERE id = ?",
                       (name, generate_password_hash(password), existing["id"]))
            uid = existing["id"]
        else:
            uid = db.execute("INSERT INTO users (name, email, password_hash, verified) VALUES (?, ?, ?, ?)",
                             (name, email, generate_password_hash(password), 0)).lastrowid
        db.commit()
        audit("signup", uid, request.remote_addr)
        session.clear()
        _issue_otp(uid, name, email, "verify")
        session["verify_uid"] = uid
        flash(f"We emailed a 6-digit code to {email}. Enter it to activate your account.", "info")
        return redirect(url_for("auth.verify_email"))
    return render_template("signup.html", form=form)


# --------------------------------------------------------------------------- email verification (OTP)
@bp.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    uid = session.get("verify_uid")
    user = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone() if uid else None
    if user is None:
        flash("Sign in or create an account first.", "info")
        return redirect(url_for("auth.login"))
    if _is_verified(user):
        return redirect(url_for("auth.login"))
    if request.method == "POST":
        db = get_db()
        if request.form.get("action") == "resend":
            err = _issue_otp(uid, user["name"], user["email"], "verify")
            flash(err or "A new code is on its way.", "error" if err else "info")
            return redirect(url_for("auth.verify_email"))
        if rate_limited("otp", 30, 900):
            flash("Too many attempts. Please wait a few minutes.", "error")
            return _render_otp("verify", user["email"], 429)
        err = _check_otp(uid, "verify", request.form.get("code", "").strip())
        if err:
            flash(err, "error")
            return _render_otp("verify", user["email"], 400)
        db.execute("UPDATE users SET verified = 1 WHERE id = ?", (uid,))
        db.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = ? WHERE id = ?",
                   (_fmt(_now()), uid))
        db.commit()
        audit("email_verified", uid, request.remote_addr)
        _start_session(uid, remember=True)
        flash(f"Email verified. Welcome to HarvestGuard, {user['name'].split()[0]}!", "success")
        return redirect(url_for("main.dashboard"))
    return _render_otp("verify", user["email"])


# --------------------------------------------------------------------------- login / logout
@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("main.dashboard"))
    nxt = request.values.get("next", "")
    if request.method == "POST":
        if rate_limited("login", 30, 600):
            flash("Too many attempts from this network. Please wait a few minutes.", "error")
            return render_template("login.html", email="", next=nxt), 429
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        generic = "Invalid email or password."
        if user is None:
            check_password_hash(_DUMMY_HASH, password)      # equalise timing, avoid user enumeration
            flash(generic, "error")
            return render_template("login.html", email=email, next=nxt), 401
        if user["locked_until"] and datetime.strptime(user["locked_until"], UTC_FMT) > _now():
            mins = int((datetime.strptime(user["locked_until"], UTC_FMT) - _now()).total_seconds() // 60) + 1
            flash(f"Too many failed attempts. This account is locked for about {mins} more minute(s).", "error")
            return render_template("login.html", email=email, next=nxt), 429
        if not check_password_hash(user["password_hash"], password):
            fails = user["failed_attempts"] + 1
            locked = None
            if fails >= current_app.config["MAX_FAILED_LOGINS"]:
                locked = _fmt(_now() + timedelta(minutes=current_app.config["LOCKOUT_MINUTES"]))
                fails = 0
                audit("account_locked", user["id"], request.remote_addr)
            db.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                       (fails, locked, user["id"]))
            db.commit()
            flash(generic if not locked else
                  f"Too many failed attempts. Account locked for {current_app.config['LOCKOUT_MINUTES']} minutes.",
                  "error")
            return render_template("login.html", email=email, next=nxt), 401
        if not _is_verified(user):          # right password, but the email was never confirmed
            session.clear()
            _issue_otp(user["id"], user["name"], user["email"], "verify")
            session["verify_uid"] = user["id"]
            flash("Please verify your email first - we sent you a new 6-digit code.", "info")
            return redirect(url_for("auth.verify_email"))
        db.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = ? WHERE id = ?",
                   (_fmt(_now()), user["id"]))
        db.commit()
        audit("login", user["id"], request.remote_addr)
        _start_session(user["id"], remember=bool(request.form.get("remember")))
        flash(f"Welcome back, {user['name'].split()[0]}.", "success")
        return redirect(nxt if is_safe_next(nxt) else url_for("main.dashboard"))
    return render_template("login.html", email="", next=nxt)


@bp.route("/logout", methods=["POST"])
def logout():
    if g.user:
        audit("logout", g.user["id"], request.remote_addr)
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("main.index"))


# --------------------------------------------------------------------------- forgot password (OTP)
@bp.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        if rate_limited("forgot", 5, 900):
            flash("Too many requests. Please wait a few minutes and try again.", "error")
            return render_template("forgot.html"), 429
        email = request.form.get("email", "").strip().lower()
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        session.clear()
        if user:
            audit("reset_requested", user["id"], request.remote_addr)
            _issue_otp(user["id"], user["name"], user["email"], "reset")
        session["reset_email"] = email        # same answer whether or not the account exists
        flash("If that email is registered, a 6-digit code has been sent.", "info")
        return redirect(url_for("auth.forgot_verify"))
    return render_template("forgot.html")


@bp.route("/forgot/verify", methods=["GET", "POST"])
def forgot_verify():
    email = session.get("reset_email")
    if not email:
        return redirect(url_for("auth.forgot"))
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if request.method == "POST":
        if request.form.get("action") == "resend":
            err = _issue_otp(user["id"], user["name"], email, "reset") if user else None
            flash(err or "If that email is registered, a new code is on its way.", "error" if err else "info")
            return redirect(url_for("auth.forgot_verify"))
        if rate_limited("otp", 30, 900):
            flash("Too many attempts. Please wait a few minutes.", "error")
            return _render_otp("reset", email, 429)
        err = _check_otp(user["id"], "reset", request.form.get("code", "").strip()) if user else \
            "That code is incorrect or has expired. Check the latest email, or request a new code."
        if err:
            flash(err, "error")
            return _render_otp("reset", email, 400)
        audit("reset_code_ok", user["id"], request.remote_addr)
        session.clear()
        session["reset_uid"] = user["id"]
        session["reset_until"] = time.time() + 600       # 10 minutes to pick the new password
        return redirect(url_for("auth.reset_password"))
    return _render_otp("reset", email)


@bp.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    uid = session.get("reset_uid")
    if not uid or session.get("reset_until", 0) < time.time():
        session.pop("reset_uid", None); session.pop("reset_until", None)
        flash("Verify the code we emailed you first, or request a new one.", "error")
        return redirect(url_for("auth.forgot"))
    if request.method == "POST":
        password, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        db = get_db()
        user = db.execute("SELECT email, name FROM users WHERE id = ?", (uid,)).fetchone()
        probs = password_problems(password, user["email"], user["name"])
        if probs or password != confirm:
            flash(("Password needs: " + ", ".join(probs) + ".") if probs else "The two passwords do not match.", "error")
            return render_template("reset.html"), 400
        db.execute("UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
                   (generate_password_hash(password), uid))
        db.execute("UPDATE users SET verified = 1 WHERE id = ?", (uid,))   # they proved they own the inbox
        db.commit()
        audit("password_reset", uid, request.remote_addr)
        session.clear()
        flash("Password updated. You can sign in now.", "success")
        return redirect(url_for("auth.login"))
    return render_template("reset.html")


# --------------------------------------------------------------------------- account
@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    db = get_db()
    uid = g.user["id"]
    if request.method == "POST":
        action = request.form.get("action")
        user = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        if action == "profile":
            name = " ".join(request.form.get("name", "").split())
            if 2 <= len(name) <= 60:
                db.execute("UPDATE users SET name = ? WHERE id = ?", (name, uid))
                db.commit()
                flash("Profile updated.", "success")
            else:
                flash("Enter your name (2-60 characters).", "error")
        elif action == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            probs = password_problems(new, user["email"], user["name"])
            if not check_password_hash(user["password_hash"], cur):
                flash("Your current password is incorrect.", "error")
            elif probs:
                flash("New password needs: " + ", ".join(probs) + ".", "error")
            elif new != conf:
                flash("The new passwords do not match.", "error")
            else:
                db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(new), uid))
                db.commit()
                audit("password_changed", uid, request.remote_addr)
                _start_session(uid, remember=True)      # rotate the session after a credential change
                flash("Password changed.", "success")
        elif action == "delete":
            if check_password_hash(user["password_hash"], request.form.get("password", "")):
                audit("account_deleted", uid, request.remote_addr)
                db.execute("DELETE FROM users WHERE id = ?", (uid,))
                db.commit()
                shutil.rmtree(Path(current_app.config["UPLOAD_DIR"]) / str(uid), ignore_errors=True)
                session.clear()
                flash("Your account and all its data were deleted.", "info")
                return redirect(url_for("main.index"))
            flash("Password incorrect - account was not deleted.", "error")
        return redirect(url_for("auth.account"))
    stats = db.execute("SELECT COUNT(*) n FROM predictions WHERE user_id = ?", (uid,)).fetchone()
    full = db.execute("SELECT created_at, last_login_at FROM users WHERE id = ?", (uid,)).fetchone()
    return render_template("account.html", n_predictions=stats["n"], user_row=full)
