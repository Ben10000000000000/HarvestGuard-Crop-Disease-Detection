import io
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402
from ml.features import FeatureConfig, extract_one  # noqa: E402
from ml.search_space import build_pipeline  # noqa: E402

CLASSES = ["Tomato___healthy", "Tomato___Late_blight", "Potato___Early_blight"]


def synth(cls: int, rng) -> np.ndarray:
    base = np.array([[40, 160, 60], [90, 70, 40], [150, 130, 50]][cls], dtype=float)
    img = base + rng.normal(0, 18, (128, 128, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    rng = np.random.default_rng(0)
    cfg = FeatureConfig(img_size=32, hsv_bins=6, use_lbp=True, lbp_radius=1, use_hog=False)
    X, y = [], []
    for c in range(3):
        for _ in range(40):
            X.append(extract_one(synth(c, rng), cfg)); y.append(c)
    params = {"classifier": "svm", "C": 5.0, "gamma_mult": 1.0, "pca": 0}
    pipe = build_pipeline(params, len(X[0]), probability=True)
    pipe.fit(np.array(X), np.array(y))
    path = tmp_path_factory.mktemp("model") / "m.joblib"
    joblib.dump({"pipeline": pipe, "classes": CLASSES, "feature_config": cfg.to_dict(), "params": params,
                 "meta": {"test_acc": 1.0, "latency_ms": 1.0, "size_kb": 10.0, "test_macro_f1": 1.0}}, path)
    return str(path)


@pytest.fixture()
def app(tmp_path, tiny_model):
    a = create_app({"TESTING": True, "TESTING_NO_RATELIMIT": True, "DATABASE": str(tmp_path / "t.db"),
                    "UPLOAD_DIR": str(tmp_path / "up"), "MODEL_PATH": tiny_model,
                    "SECRET_KEY": "test", "SHOW_OTP": True, "OTP_RESEND_SECONDS": 0})
    from app import db
    with a.app_context():
        db.init_db()
    Path(a.config["UPLOAD_DIR"]).mkdir(exist_ok=True)
    return a


@pytest.fixture()
def client(app):
    return app.test_client()


def token(client, path="/forgot"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def signup_raw(client, name="Asha Rao", email="asha@example.com", password="Sunrise2024!"):
    t = token(client, "/signup")
    return client.post("/signup", data={"csrf_token": t, "name": name, "email": email, "password": password,
                                        "confirm": password}, follow_redirects=False)


def read_otp(client, path):
    """The demo-mode page shows the code (SHOW_OTP) - read it like a user reads their inbox."""
    return re.search(r'class="otp-demo">(\d{6})<', client.get(path).get_data(as_text=True)).group(1)


def signup(client, name="Asha Rao", email="asha@example.com", password="Sunrise2024!"):
    """Sign up AND confirm the e-mail OTP, leaving the client logged in (what most tests need)."""
    r = signup_raw(client, name, email, password)
    if r.status_code != 302:
        return r
    code = read_otp(client, "/verify-email")
    return client.post("/verify-email", data={"csrf_token": token(client, "/verify-email"), "action": "verify", "code": code})


def login(client, email="asha@example.com", password="Sunrise2024!", **extra):
    t = token(client, "/login")
    return client.post("/login", data={"csrf_token": t, "email": email, "password": password, **extra})


def png_bytes(cls=1, seed=1):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO(); Image.fromarray(synth(cls, rng)).save(buf, format="PNG"); return buf.getvalue()
