"""PlantVillage loading, caching and splitting.

Expected layout (Kaggle "plantvillage-dataset" or the original GitHub repo):

    <data-dir>/<ClassName>/<image>.jpg           (or .../color/<ClassName>/<image>.jpg)

Images are centre-cropped, resized once to 128x128 and cached in a single .npz.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from .features import BASE_SIZE, load_image_base

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_root(data_dir: Path) -> Path:
    """Locate the folder whose sub-folders are the class folders."""
    data_dir = Path(data_dir)
    for d in [data_dir, *sorted(p for p in data_dir.rglob("*") if p.is_dir())]:
        if d.name.lower() == "color":
            return d
    return data_dir


def list_images(root: Path, max_per_class: int | None, seed: int = 0):
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    for c in classes:
        files = sorted(f for f in (root / c).iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            continue
        if max_per_class and len(files) > max_per_class:
            idx = rng.choice(len(files), max_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        paths += files
        labels += [c] * len(files)
    classes = sorted(set(labels))
    return paths, labels, classes


def load_dataset(data_dir, max_per_class: int | None = None, cache_dir="data", n_jobs=2, seed=0):
    root = find_class_root(Path(data_dir))
    paths, labels, classes = list_images(root, max_per_class, seed)
    if not paths:
        raise SystemExit(f"No images found under {root}. Expected <class>/<image>.jpg folders.")
    key = hashlib.md5(("|".join(str(p) for p in paths) + str(BASE_SIZE)).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"plantvillage_{len(paths)}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], json.loads(str(z["classes"]))
    print(f"[data] loading {len(paths)} images from {root} ...", flush=True)
    imgs = Parallel(n_jobs=n_jobs)(delayed(load_image_base)(p) for p in paths)
    X = np.stack(imgs).astype(np.uint8)
    y = np.array([classes.index(l) for l in labels], dtype=np.int64)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, classes=json.dumps(classes))
    return X, y, classes


def split(X, y, seed=42, val=0.15, test=0.15):
    """Stratified train / validation / test split (validation is used by WOA, test only once)."""
    idx = np.arange(len(y))
    tr_va, te = train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    tr, va = train_test_split(tr_va, test_size=val / (1 - test), stratify=y[tr_va], random_state=seed)
    return tr, va, te
