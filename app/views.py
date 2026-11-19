"""Public pages, leaf detection, history and the optimisation report."""
from __future__ import annotations

import json
from pathlib import Path

from flask import (Blueprint, abort, current_app, flash, g, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)

from . import advisor, inference
from .db import get_db
from .security import login_required, rate_limited

bp = Blueprint("main", __name__)


def _model_path() -> str:
    return current_app.config["MODEL_PATH"]


def _meta() -> dict | None:
    try:
        return inference.load_model(_model_path())["meta"]
    except FileNotFoundError:
        return None


def _results() -> dict | None:
    f = Path(current_app.config["REPORT_DIR"]) / "results.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            return None
    return None


@bp.app_context_processor
def inject():
    return {"model_meta": _meta()}


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/healthz")
def healthz():
    return jsonify(status="ok", model_loaded=_meta() is not None)


# --------------------------------------------------------------------------- dashboard
@bp.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    uid = g.user["id"]
    totals = db.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(is_healthy),0) healthy, COALESCE(AVG(confidence),0) conf "
        "FROM predictions WHERE user_id = ?", (uid,)).fetchone()
    top = db.execute(
        "SELECT condition, crop, COUNT(*) n FROM predictions WHERE user_id = ? AND is_healthy = 0 "
        "GROUP BY label ORDER BY n DESC LIMIT 5", (uid,)).fetchall()
    recent = db.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 6", (uid,)).fetchall()
    return render_template("dashboard.html", totals=totals, top=top, recent=recent)


# --------------------------------------------------------------------------- detection
def _run_detection(file_storage, note: str = ""):
    data = file_storage.read()
    if not data:
        raise inference.BadImage("Please choose a leaf photo first.")
    img = inference.read_image(data)
    result = inference.classify(img, _model_path())
    rel = inference.save_upload(img, current_app.config["UPLOAD_DIR"], g.user["id"])
    crop, cond, healthy = advisor.parse_label(result["label"])
    db = get_db()
    cur = db.execute(
        "INSERT INTO predictions (user_id, image_file, label, crop, condition, is_healthy, confidence, "
        "top3_json, latency_ms, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (g.user["id"], rel, result["label"], crop, cond, int(healthy), result["confidence"],
         json.dumps(result["top"]), result["latency_ms"], note[:200]))
    db.commit()
    return cur.lastrowid, result


@bp.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    if request.method == "POST":
        if rate_limited(f"detect{g.user['id']}", 60, 60):
            flash("You're scanning very fast - wait a moment and try again.", "error")
            return render_template("detect.html", ready=_meta() is not None), 429
        f = request.files.get("leaf")
        if f is None:
            flash("Please choose a leaf photo first.", "error")
            return render_template("detect.html", ready=_meta() is not None), 400
        try:
            pid, _ = _run_detection(f, request.form.get("note", ""))
        except inference.BadImage as e:
            flash(str(e), "error")
            return render_template("detect.html", ready=_meta() is not None), 400
        except FileNotFoundError as e:
            flash(str(e), "error")
            return render_template("detect.html", ready=False), 503
        return redirect(url_for("main.result", pid=pid))
    return render_template("detect.html", ready=_meta() is not None)


@bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """JSON endpoint (same session + X-CSRF-Token header as the web UI)."""
    f = request.files.get("leaf")
    if f is None:
        return jsonify(error="Send the photo as multipart field 'leaf'."), 400
    try:
        pid, res = _run_detection(f, request.form.get("note", ""))
    except inference.BadImage as e:
        return jsonify(error=str(e)), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    adv = advisor.advice_for(res["label"])
    return jsonify(id=pid, label=res["label"], crop=adv["crop"], condition=adv["condition"],
                   healthy=adv["healthy"], confidence=round(res["confidence"], 4),
                   uncertain=res["confidence"] < current_app.config["CONFIDENCE_THRESHOLD"],
                   top=res["top"], latency_ms=round(res["latency_ms"], 1), advice=adv["actions"])


def _owned_prediction(pid: int):
    row = get_db().execute("SELECT * FROM predictions WHERE id = ? AND user_id = ?", (pid, g.user["id"])).fetchone()
    if row is None:
        abort(404)
    return row


@bp.route("/history/<int:pid>")
@login_required
def result(pid):
    row = _owned_prediction(pid)
    top = json.loads(row["top3_json"])
    for t in top:
        t["crop"], t["condition"], t["healthy"] = advisor.parse_label(t["label"])
    return render_template("result.html", p=row, top=top, advice=advisor.advice_for(row["label"]),
                           uncertain=row["confidence"] < current_app.config["CONFIDENCE_THRESHOLD"])


@bp.route("/history")
@login_required
def history():
    q = request.args.get("filter", "all")
    page = max(request.args.get("page", 1, type=int), 1)
    per = 12
    where, args = "WHERE user_id = ?", [g.user["id"]]
    if q == "healthy":
        where += " AND is_healthy = 1"
    elif q == "diseased":
        where += " AND is_healthy = 0"
    db = get_db()
    total = db.execute(f"SELECT COUNT(*) n FROM predictions {where}", args).fetchone()["n"]
    rows = db.execute(f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                      args + [per, (page - 1) * per]).fetchall()
    return render_template("history.html", rows=rows, q=q, page=page, pages=max((total + per - 1) // per, 1),
                           total=total)


@bp.route("/history/<int:pid>/delete", methods=["POST"])
@login_required
def delete_prediction(pid):
    row = _owned_prediction(pid)
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id = ?", (pid,))
    db.commit()
    (Path(current_app.config["UPLOAD_DIR"]) / row["image_file"]).unlink(missing_ok=True)
    flash("Scan deleted.", "info")
    return redirect(url_for("main.history"))


@bp.route("/media/<int:pid>")
@login_required
def media(pid):
    row = _owned_prediction(pid)          # only the owner can ever fetch an image
    return send_from_directory(current_app.config["UPLOAD_DIR"], row["image_file"], mimetype="image/jpeg")


# --------------------------------------------------------------------------- optimisation report
@bp.route("/model")
def model_report():
    return render_template("model.html", results=_results())


@bp.route("/reports/<path:name>")
def report_file(name):
    if not name.lower().endswith(".png"):
        abort(404)
    return send_from_directory(current_app.config["REPORT_DIR"], name, max_age=3600)
