import re
from conftest import login, read_otp, signup, signup_raw, token


def test_signup_logs_in_and_redirects(client):
    r = signup(client)
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    assert b"Hello, Asha" in client.get("/dashboard").data


def test_password_is_hashed(client, app):
    signup(client)
    from app.db import get_db
    with app.app_context():
        h = get_db().execute("SELECT password_hash FROM users").fetchone()[0]
    assert "Sunrise2024!" not in h and h.split(":")[0] in ("scrypt", "pbkdf2")


def test_signup_validation(client):
    assert signup(client, password="short1").status_code == 400
    assert signup(client, password="allletters").status_code == 400
    assert signup(client, password="password123").status_code == 400
    assert signup(client, email="not-an-email").status_code == 400
    t = token(client, "/signup")
    r = client.post("/signup", data={"csrf_token": t, "name": "A B", "email": "a@b.co", "password": "Sunrise2024!", "confirm": "Different1!"})
    assert r.status_code == 400


def test_duplicate_email_case_insensitive(client):
    signup(client); client.post("/logout", data={"csrf_token": token(client)})
    r = signup(client, email="ASHA@Example.com")
    assert r.status_code == 400 and b"already exists" in r.data


def test_csrf_required(client):
    r = client.post("/login", data={"email": "a@b.co", "password": "x"})
    assert r.status_code == 400


def test_login_wrong_password_generic_message(client):
    signup(client); client.post("/logout", data={"csrf_token": token(client)})
    r = login(client, password="WrongPass1")
    r2 = login(client, email="nobody@example.com")
    assert r.status_code == r2.status_code == 401
    assert b"Invalid email or password" in r.data and b"Invalid email or password" in r2.data


def test_lockout_after_repeated_failures(client):
    signup(client); client.post("/logout", data={"csrf_token": token(client)})
    for _ in range(5):
        login(client, password="WrongPass1")
    r = login(client)               # correct password, but locked
    assert r.status_code == 429 and b"locked" in r.data


def test_login_success_and_logout(client):
    signup(client); client.post("/logout", data={"csrf_token": token(client)})
    r = login(client)
    assert r.status_code == 302
    assert client.get("/dashboard").status_code == 200
    client.post("/logout", data={"csrf_token": token(client, "/dashboard")})
    assert client.get("/dashboard").status_code == 302


def test_protected_routes_redirect_to_login(client):
    for path in ("/dashboard", "/detect", "/history", "/account", "/history/1", "/media/1"):
        r = client.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"], path


def test_open_redirect_blocked(client):
    signup(client); client.post("/logout", data={"csrf_token": token(client)})
    r = login(client, next="https://evil.example/steal")
    assert r.headers["Location"].endswith("/dashboard")
    client.post("/logout", data={"csrf_token": token(client, "/dashboard")})
    r = login(client, next="/history")
    assert r.headers["Location"].endswith("/history")


def _post(client, path, **data):
    return client.post(path, data={"csrf_token": token(client, path), **data})


def test_signup_requires_otp_before_login(client):
    r = signup_raw(client)
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify-email")
    assert client.get("/dashboard").status_code == 302                      # not signed in yet
    assert _post(client, "/verify-email", action="verify", code="000000").status_code == 400
    code = read_otp(client, "/verify-email")
    r = _post(client, "/verify-email", action="verify", code=code)
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    assert b"Hello, Asha" in client.get("/dashboard").data


def test_unverified_user_cannot_login_until_verified(client):
    signup_raw(client)
    r = login(client)                                                       # correct password
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify-email")
    assert client.get("/dashboard").status_code == 302
    _post(client, "/verify-email", action="verify", code=read_otp(client, "/verify-email"))
    assert client.get("/dashboard").status_code == 200


def test_otp_single_use_and_attempt_limit(client, app):
    signup_raw(client)
    code = read_otp(client, "/verify-email")
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(5):
        assert _post(client, "/verify-email", action="verify", code=wrong).status_code == 400
    r = _post(client, "/verify-email", action="verify", code=code)          # burned by 5 wrong guesses
    assert r.status_code == 400 and b"Too many wrong attempts" in r.data
    _post(client, "/verify-email", action="resend")
    code2 = read_otp(client, "/verify-email")
    assert _post(client, "/verify-email", action="verify", code=code2).status_code == 302
    client.post("/logout", data={"csrf_token": token(client, "/dashboard")})
    assert login(client).status_code == 302


def test_otp_expires(client, app):
    signup_raw(client)
    code = read_otp(client, "/verify-email")
    import sqlite3
    con = sqlite3.connect(app.config["DATABASE"])
    con.execute("UPDATE otps SET expires_at = '2000-01-01 00:00:00'"); con.commit(); con.close()
    assert _post(client, "/verify-email", action="verify", code=code).status_code == 400


def test_resend_cooldown(client, app):
    app.config["OTP_RESEND_SECONDS"] = 60
    signup_raw(client)
    r = client.post("/verify-email", data={"csrf_token": token(client, "/verify-email"), "action": "resend"}, follow_redirects=True)
    assert b"Please wait" in r.data


def test_reverify_after_unconfirmed_signup_takes_over_account(client):
    signup_raw(client, name="Attacker", password="Attack2024!x")
    client.get("/logout")
    r = signup_raw(client, name="Real Owner", password="Sunrise2024!")      # real owner signs up later
    assert r.status_code == 302
    _post(client, "/verify-email", action="verify", code=read_otp(client, "/verify-email"))
    client.post("/logout", data={"csrf_token": token(client, "/dashboard")})
    assert login(client, password="Attack2024!x").status_code == 401
    assert login(client, password="Sunrise2024!").status_code == 302


def test_forgot_password_otp_flow(client):
    signup(client); client.post("/logout", data={"csrf_token": token(client, "/dashboard")})
    r = _post(client, "/forgot", email="asha@example.com")
    assert r.status_code == 302 and r.headers["Location"].endswith("/forgot/verify")
    assert client.get("/reset-password").status_code == 302                  # no code yet -> refused
    assert _post(client, "/forgot/verify", action="verify", code="123456").status_code == 400
    code = read_otp(client, "/forgot/verify")
    r = _post(client, "/forgot/verify", action="verify", code=code)
    assert r.status_code == 302 and r.headers["Location"].endswith("/reset-password")
    assert _post(client, "/reset-password", password="weak", confirm="weak").status_code == 400
    r = _post(client, "/reset-password", password="NewHarvest77", confirm="NewHarvest77")
    assert r.status_code == 302
    assert client.get("/reset-password").status_code == 302                  # session consumed
    assert login(client, password="Sunrise2024!").status_code == 401
    assert login(client, password="NewHarvest77").status_code == 302


def test_forgot_does_not_reveal_unknown_email(client):
    r = _post(client, "/forgot", email="ghost@example.com")
    assert r.status_code == 302 and r.headers["Location"].endswith("/forgot/verify")
    page = client.get("/forgot/verify").get_data(as_text=True)
    assert "otp-demo" not in page                                           # no code issued
    assert _post(client, "/forgot/verify", action="verify", code="123456").status_code == 400


def test_change_password_and_delete_account(client):
    signup(client)
    t = token(client, "/account")
    r = client.post("/account", data={"csrf_token": t, "action": "password", "current": "Sunrise2024!", "new": "Another2025x", "confirm": "Another2025x"}, follow_redirects=True)
    assert b"Password changed" in r.data
    t = token(client, "/account")
    r = client.post("/account", data={"csrf_token": t, "action": "delete", "password": "bad"}, follow_redirects=True)
    assert b"not deleted" in r.data
    t = token(client, "/account")
    client.post("/account", data={"csrf_token": t, "action": "delete", "password": "Another2025x"})
    assert login(client, password="Another2025x").status_code == 401
