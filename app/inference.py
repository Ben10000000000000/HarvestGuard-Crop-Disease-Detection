"""Loads the WOA-tuned model once and classifies a leaf photo."""
from __future__ import annotations

import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from ml.features import FeatureConfig, extract_one, load_image_base

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_PIXELS = 40_000_000

_lock = threading.Lock()
_cache: dict[str, dict] = {}


class BadImage(ValueError):
    pass


def load_model(path: str) -> dict:
    with _lock:
        if path not in _cache:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Model file not found at {path}. Run `python -m ml.train ...` first (see README).")
            bundle = joblib.load(path)
            bundle["cfg"] = FeatureConfig.from_dict(bundle["feature_config"])
            # single-threaded inference = predictable latency on small devices
            clf = bundle["pipeline"].named_steps["clf"]
            if hasattr(clf, "n_jobs"):
                clf.set_params(n_jobs=1)
            _cache[path] = bundle
        return _cache[path]


def read_image(data: bytes) -> Image.Image:
    """Validate an upload and return a clean RGB PIL image (EXIF / metadata dropped)."""
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        probe = Image.open(BytesIO(data))
        fmt = probe.format
        probe.verify()
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError, ValueError):
        raise BadImage("That file isn't a readable image. Please upload a JPG, PNG or WebP photo of a leaf.")
    if fmt not in ALLOWED_FORMATS:
        raise BadImage("Unsupported image type. Please upload a JPG, PNG or WebP photo.")
    if min(img.size) < 32:
        raise BadImage("The image is too small - use a photo at least 64x64 pixels.")
    return img.convert("RGB")


def classify(img: Image.Image, model_path: str, top_k: int = 3) -> dict:
    bundle = load_model(model_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    t0 = time.perf_counter()
    arr = load_image_base(buf)
    feats = extract_one(arr, bundle["cfg"])[None]
    proba = bundle["pipeline"].predict_proba(feats)[0]
    latency = (time.perf_counter() - t0) * 1000
    classes = bundle["classes"]
    order = np.argsort(proba)[::-1][:top_k]
    top = [{"label": classes[i], "confidence": float(proba[i])} for i in order]
    return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top": top,
            "latency_ms": latency}


def save_upload(img: Image.Image, upload_dir: str, user_id: int) -> str:
    """Store a re-encoded (<=640px) JPEG copy; returns the path relative to upload_dir."""
    img = img.copy()
    img.thumbnail((640, 640))
    rel = f"{user_id}/{uuid.uuid4().hex}.jpg"
    dest = Path(upload_dir) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=88)
    return rel
