import io
from pathlib import Path

from conftest import login, png_bytes, signup, token


def upload(client, data, name="leaf.png"):
    t = token(client, "/detect")
    return client.post("/detect", data={"csrf_token": t, "leaf": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_detect_saves_result_and_shows_advice(client):
    signup(client)
    r = upload(client, png_bytes(cls=1))
    assert r.status_code == 302
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Late blight" in page and "What to do" in page
    assert "History" in client.get("/history").get_data(as_text=True)


def test_rejects_non_images_and_empty(client):
    signup(client)
    assert upload(client, b"not an image at all", "x.png").status_code == 400
    assert upload(client, b"", "x.png").status_code == 400
    assert upload(client, b"<?php echo 1;?>", "shell.php").status_code == 400


def test_api_predict_json(client):
    signup(client)
    t = token(client, "/detect")
    r = client.post("/api/predict", data={"leaf": (io.BytesIO(png_bytes(0)), "a.png")}, headers={"X-CSRF-Token": t},
                    content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["label"] == "Tomato___healthy" and j["healthy"] is True and len(j["top"]) == 3


def test_api_requires_login_and_csrf(client):
    r = client.post("/api/predict", data={"leaf": (io.BytesIO(png_bytes()), "a.png")}, content_type="multipart/form-data")
    assert r.status_code in (302, 400)


def test_users_cannot_see_each_others_scans(app):
    a, b = app.test_client(), app.test_client()
    signup(a, email="a@example.com"); signup(b, email="b@example.com", name="Bo Lin")
    loc = upload(a, png_bytes()).headers["Location"]
    assert a.get(loc).status_code == 200
    assert b.get(loc).status_code == 404
    pid = loc.rsplit("/", 1)[1]
    assert b.get(f"/media/{pid}").status_code == 404
    t = token(b, "/history")
    assert b.post(f"/history/{pid}/delete", data={"csrf_token": t}).status_code == 404


def test_delete_scan_removes_file(client, app):
    signup(client)
    loc = upload(client, png_bytes()).headers["Location"]
    files = list(Path(app.config["UPLOAD_DIR"]).rglob("*.jpg")); assert len(files) == 1
    t = token(client, "/history")
    client.post(loc + "/delete", data={"csrf_token": t})
    assert not list(Path(app.config["UPLOAD_DIR"]).rglob("*.jpg"))


def test_security_headers_and_healthz(client):
    r = client.get("/healthz")
    assert r.get_json()["model_loaded"] is True
    assert "default-src 'self'" in r.headers["Content-Security-Policy"] and r.headers["X-Frame-Options"] == "DENY"
