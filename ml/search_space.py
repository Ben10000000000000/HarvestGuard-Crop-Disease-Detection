"""The search space the Whale Optimization Algorithm explores.

A whale is a vector of 12 genes in [0, 1]. `decode` turns it into a concrete
configuration (feature extractor + classifier hyper-parameters), `build_pipeline`
turns that configuration into a scikit-learn Pipeline.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FeatureConfig

IMG_SIZES = [32, 48, 64, 96]
HSV_BINS = [4, 6, 8, 12, 16]
HOG_PPC = [8, 16]
PCA_CHOICES = [0, 16, 32, 64, 96]
CLASSIFIERS = ["svm", "mlp", "rf"]
GENE_NAMES = ["img_size", "hsv_bins", "use_lbp", "lbp_radius", "use_hog", "hog_ppc",
              "use_jhist", "use_glcm", "pca", "classifier", "hp1", "hp2"]
DIM = len(GENE_NAMES)


def _pick(g: float, options: list):
    return options[min(int(g * len(options)), len(options) - 1)]


def decode(genes) -> dict:
    g = np.clip(np.asarray(genes, dtype=float), 0.0, 1.0)
    clf = _pick(g[9], CLASSIFIERS)
    p = {
        "img_size": _pick(g[0], IMG_SIZES),
        "hsv_bins": _pick(g[1], HSV_BINS),
        "use_lbp": bool(g[2] >= 0.5),
        "lbp_radius": 1 + min(int(g[3] * 3), 2),
        "use_hog": bool(g[4] >= 0.5),
        "hog_ppc": _pick(g[5], HOG_PPC),
        "use_jhist": bool(g[6] >= 0.5),
        "use_glcm": bool(g[7] >= 0.5),
        "pca": _pick(g[8], PCA_CHOICES),
        "classifier": clf,
    }
    if clf == "svm":
        p["C"] = round(float(10 ** (-1 + 3 * g[10])), 4)              # 0.1 .. 100
        p["gamma_mult"] = round(float(10 ** (-1.5 + 3 * g[11])), 4)   # gamma = mult / n_features
    elif clf == "mlp":
        p["hidden"] = 16 + int(g[10] * 112)                           # 16 .. 128 units
        p["alpha"] = round(float(10 ** (-5 + 3 * g[11])), 6)          # L2 strength
    else:
        p["n_estimators"] = 20 + int(g[10] * 130)                     # 20 .. 150
        p["max_depth"] = 8 + int(g[11] * 24)                          # 8 .. 32
    return p


def canonical(p: dict) -> dict:
    """Drop parameters that have no effect so equal configs share one cache entry."""
    q = dict(p)
    if not q["use_lbp"]:
        q["lbp_radius"] = 1
    if not q["use_hog"]:
        q["hog_ppc"] = 16
    return q


def feature_config(p: dict) -> FeatureConfig:
    return FeatureConfig(img_size=p["img_size"], hsv_bins=p["hsv_bins"], use_lbp=p["use_lbp"],
                         lbp_radius=p["lbp_radius"], use_hog=p["use_hog"], hog_ppc=p["hog_ppc"],
                         use_jhist=p["use_jhist"], use_glcm=p["use_glcm"])


def build_pipeline(p: dict, n_features: int, probability: bool = False, seed: int = 42,
                   n_jobs: int = 2) -> Pipeline:
    steps = [("scale", StandardScaler())]
    n_out = n_features
    k = p.get("pca", 0)
    if k and k < n_features - 1:
        steps.append(("pca", PCA(n_components=k, random_state=seed)))
        n_out = k
    c = p["classifier"]
    warnings.filterwarnings("ignore", message=".*probability.*deprecated.*", category=FutureWarning)
    if c == "svm":
        clf = SVC(kernel="rbf", C=p["C"], gamma=p["gamma_mult"] / n_out, probability=probability,
                  class_weight="balanced", random_state=seed)
    elif c == "mlp":
        # small batches + higher LR + patience: sklearn's defaults stop far too early on 38-class data
        clf = MLPClassifier(hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"], max_iter=400,
                            batch_size=32, learning_rate_init=3e-3, early_stopping=True,
                            n_iter_no_change=15, random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     class_weight="balanced", n_jobs=n_jobs, random_state=seed)
    steps.append(("clf", clf))
    return Pipeline(steps)


def encode_default() -> np.ndarray:
    """Gene vector that decodes to a sensible hand-tuned configuration (used as baseline)."""
    g = np.full(DIM, 0.5)
    g[0] = 0.55   # 64 px
    g[1] = 0.45   # 8 bins
    g[2] = 0.75   # LBP on
    g[3] = 0.0    # radius 1
    g[4] = 0.75   # HOG on
    g[5] = 0.75   # ppc 16
    g[6] = 0.0    # joint histogram off
    g[7] = 0.0    # GLCM off
    g[8] = 0.0    # no PCA
    g[9] = 0.1    # SVM
    g[10] = 0.5   # C ~ 10^0.5
    g[11] = 0.5   # gamma_mult ~ 1
    return g
