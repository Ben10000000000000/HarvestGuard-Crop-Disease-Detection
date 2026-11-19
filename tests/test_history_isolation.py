"""A signed-in user must only ever see / open / delete their OWN scans."""
import io
from pathlib import Path

from conftest import png_bytes, signup, token


def _scan(client, cls):
    t = token(client, "/detect")
    r = client.post("/detect", data={"csrf_token": t, "leaf": (io.BytesIO(png_bytes(cls=cls)), "leaf.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 302
    return int(r.headers["Location"].rstrip("/").split("/")[-1])


def test_history_is_private_per_user(app):
    alice, bob = app.test_client(), app.test_client()
    signup(alice, "Alice A", "alice@example.com")
    signup(bob, "Bob B", "bob@example.com")
    a_id = _scan(alice, 1)
    b_id = _scan(bob, 0)

    # each history lists only its owner's scan
    assert f"/history/{a_id}" in alice.get("/history").get_data(as_text=True)
    assert f"/history/{b_id}" not in alice.get("/history").get_data(as_text=True)
    assert f"/history/{b_id}" in bob.get("/history").get_data(as_text=True)
    assert f"/history/{a_id}" not in bob.get("/history").get_data(as_text=True)

    # direct URL guessing is refused for result page, image and delete
    assert alice.get(f"/history/{b_id}").status_code == 404
    assert alice.get(f"/media/{b_id}").status_code == 404
    t = token(alice, "/history")
    assert alice.post(f"/history/{b_id}/delete", data={"csrf_token": t}).status_code == 404
    assert bob.get(f"/history/{b_id}").status_code == 200  # still there

    # logged-out visitors get nothing
    anon = app.test_client()
    assert anon.get("/history").status_code == 302
    assert anon.get(f"/media/{a_id}").status_code in (302, 401, 404)
