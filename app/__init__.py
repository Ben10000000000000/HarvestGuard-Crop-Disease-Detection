"""HarvestGuard Flask application factory."""
from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask, render_template

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path):
    """Tiny .env reader (KEY=VALUE per line) so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _secret_key(instance: Path) -> str:
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    f = instance / "secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
        try:
            f.chmod(0o600)
        except OSError:
            pass
    return f.read_text().strip()


def create_app(test_config: dict | None = None) -> Flask:
    _load_dotenv(ROOT / ".env")
    instance = Path(os.environ.get("HG_INSTANCE_DIR", ROOT / "instance"))
    instance.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, instance_path=str(instance))
    app.config.update(
        SECRET_KEY=_secret_key(instance),
        DATABASE=str(instance / "harvestguard.sqlite3"),  # tests only; production uses MongoDB
        MONGO_URI=os.environ.get("HG_MONGO_URI", "mongodb://localhost:27017"),
        MONGO_DB=os.environ.get("HG_MONGO_DB", "harvestguard_db"),
        UPLOAD_DIR=str(instance / "uploads"),
        MODEL_PATH=os.environ.get("HG_MODEL_PATH", str(ROOT / "models" / "harvestguard_model.joblib")),
        REPORT_DIR=str(ROOT / "reports"),
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("HG_COOKIE_SECURE", "0") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        # authentication policy
        MAX_FAILED_LOGINS=5,
        LOCKOUT_MINUTES=10,
        OTP_MINUTES=10,            # how long an e-mailed code stays valid
        OTP_MAX_ATTEMPTS=5,        # wrong guesses before the code is burned
        OTP_RESEND_SECONDS=60,     # minimum gap between two codes for the same purpose
        # no SMTP: the code goes to the server console; HG_SHOW_OTP=1 also shows it on screen (local demo ONLY)
        SHOW_OTP=os.environ.get("HG_SHOW_OTP", os.environ.get("HG_SHOW_RESET_LINK", "0")) == "1",
        CONFIDENCE_THRESHOLD=0.55,
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    from . import db, security
    db.init_app(app)
    security.init_app(app)

    from .auth import bp as auth_bp
    from .views import bp as main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    @app.errorhandler(400)
    def _400(e):
        return render_template("error.html", code=400, message=getattr(e, "description", "Bad request.")), 400

    @app.errorhandler(404)
    def _404(e):
        return render_template("error.html", code=404, message="That page doesn't exist."), 404

    @app.errorhandler(413)
    def _413(e):
        return render_template("error.html", code=413, message="That image is too large (max 8 MB)."), 413

    @app.errorhandler(403)
    def _403(e):
        return render_template("error.html", code=403, message="You don't have access to that."), 403

    return app
