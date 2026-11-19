"""Hand-crafted, cheap-to-compute leaf features.

Everything here is intentionally light-weight (colour statistics, LBP texture, HOG)
so the whole pipeline can run on a Raspberry-Pi class device without a GPU.
The Whale Optimization Algorithm decides *which* of these blocks are worth their
compute cost (see ml/search_space.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

BASE_SIZE = 128  # every image is cached once at 128x128 and down-scaled per config


@dataclass(frozen=True)
class FeatureConfig:
    img_size: int = 64
    hsv_bins: int = 8
    use_lbp: bool = True
    lbp_radius: int = 1
    use_hog: bool = False
    hog_ppc: int = 16
    use_jhist: bool = False
    use_glcm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FeatureConfig":
        return FeatureConfig(**{k: d[k] for k in FeatureConfig.__dataclass_fields__})


# --------------------------------------------------------------------------- loading
def load_image_base(source) -> np.ndarray:
    """Open a path / file object, centre-crop to a square and resize to BASE_SIZE (RGB uint8)."""
    img = Image.open(source)
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    arr = np.asarray(img)
    return cv2.resize(arr, (BASE_SIZE, BASE_SIZE), interpolation=cv2.INTER_AREA)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if size == img.shape[0]:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- blocks
def base_block(img: np.ndarray, size: int, bins: int) -> np.ndarray:
    """Colour moments (HSV+Lab), lesion-colour ratios and HSV histograms."""
    im = _resize(img, size)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    feats = []
    for space in (hsv, lab):
        flat = space.reshape(-1, 3).astype(np.float32) / 255.0
        m = flat.mean(0)
        s = flat.std(0)
        sk = np.cbrt(((flat - m) ** 3).mean(0))
        feats += [m, s, sk]
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = s_ > 40
    ratios = [
        np.mean(sat & (h >= 35) & (h <= 85)),   # healthy green
        np.mean(sat & (h >= 18) & (h < 35)),    # yellowing / chlorosis
        np.mean(sat & (h < 18) & (v < 200)),    # brown / necrotic
        np.mean(v < 60),                        # very dark (black rot, mould)
        np.mean((s_ < 30) & (v > 180)),         # whitish (powdery mildew)
    ]
    feats.append(np.asarray(ratios, dtype=np.float32))
    for ch, rng in zip(range(3), ((0, 180), (0, 256), (0, 256))):
        hist = cv2.calcHist([hsv], [ch], None, [bins], list(rng)).ravel()
        feats.append((hist / max(hist.sum(), 1.0)).astype(np.float32))
    return np.concatenate(feats).astype(np.float32)


def lbp_block(img: np.ndarray, size: int, radius: int) -> np.ndarray:
    gray = cv2.cvtColor(_resize(img, size), cv2.COLOR_RGB2GRAY)
    p = 8 * radius
    lbp = local_binary_pattern(gray, p, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def hog_block(img: np.ndarray, size: int, ppc: int) -> np.ndarray:
    gray = cv2.cvtColor(_resize(img, size), cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2),
               feature_vector=True).astype(np.float32)


def jhist_block(img: np.ndarray, size: int, hue_bins: int) -> np.ndarray:
    """Joint HSV histogram (hue_bins x 4 x 4): captures colour *combinations* (e.g. brown on green)."""
    hsv = cv2.cvtColor(_resize(img, size), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [hue_bins, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (h / max(h.sum(), 1.0)).astype(np.float32)


def glcm_block(img: np.ndarray, size: int, levels: int = 16) -> np.ndarray:
    """Grey-level co-occurrence texture statistics (contrast, homogeneity, ...)."""
    gray = cv2.cvtColor(_resize(img, size), cv2.COLOR_RGB2GRAY)
    q = (gray.astype(np.int32) * levels // 256).astype(np.uint8)
    m = graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels,
                     symmetric=True, normed=True)
    props = ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")
    return np.concatenate([graycoprops(m, p).ravel() for p in props]).astype(np.float32)


def block_keys(cfg: FeatureConfig) -> list[tuple]:
    keys = [("base", cfg.img_size, cfg.hsv_bins)]
    if cfg.use_lbp:
        keys.append(("lbp", cfg.img_size, cfg.lbp_radius))
    if cfg.use_hog:
        keys.append(("hog", cfg.img_size, cfg.hog_ppc))
    if cfg.use_jhist:
        keys.append(("jhist", cfg.img_size, 8))
    if cfg.use_glcm:
        keys.append(("glcm", cfg.img_size, 16))
    return keys


def compute_block(img: np.ndarray, key: tuple) -> np.ndarray:
    kind, size, param = key
    if kind == "base":
        return base_block(img, size, param)
    if kind == "lbp":
        return lbp_block(img, size, param)
    if kind == "hog":
        return hog_block(img, size, param)
    if kind == "jhist":
        return jhist_block(img, size, param)
    if kind == "glcm":
        return glcm_block(img, size, param)
    raise ValueError(key)


def extract_one(img: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Feature vector for a single 128x128 RGB image (used at inference time)."""
    return np.concatenate([compute_block(img, k) for k in block_keys(cfg)]).astype(np.float32)
