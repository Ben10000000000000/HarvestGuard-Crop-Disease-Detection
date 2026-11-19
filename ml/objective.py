"""Fitness function: accuracy minus a penalty for compute cost on the target device.

    fitness = val_accuracy - lam_latency * (latency_ms / latency_budget_ms)
                           - lam_size    * (model_kb   / size_budget_kb)

Latency is measured end-to-end for ONE image (feature extraction + prediction, single
thread) and model size is the pickled pipeline size, so the optimiser is rewarded for
configurations that stay accurate *and* cheap enough for a field device.
"""
from __future__ import annotations

import pickle
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score

from .features import FeatureConfig, block_keys, compute_block, extract_one
from .search_space import build_pipeline, canonical, feature_config


class FeatureStore:
    """Computes feature blocks once per (block, size, param) and reuses them."""

    def __init__(self, images: np.ndarray, n_jobs: int = 2):
        self.images, self.n_jobs, self.cache = images, n_jobs, {}

    def block(self, key) -> np.ndarray:
        if key not in self.cache:
            chunks = np.array_split(np.arange(len(self.images)), max(self.n_jobs * 4, 1))
            parts = Parallel(n_jobs=self.n_jobs)(
                delayed(lambda ids: np.stack([compute_block(self.images[i], key) for i in ids]))(c)
                for c in chunks if len(c))
            self.cache[key] = np.concatenate(parts)
        return self.cache[key]

    def matrix(self, cfg: FeatureConfig, idx: np.ndarray) -> np.ndarray:
        return np.hstack([self.block(k)[idx] for k in block_keys(cfg)])


class Evaluator:
    def __init__(self, store: FeatureStore, y, tr, va, lam_lat=0.02, lam_size=0.02,
                 lat_budget_ms=25.0, size_budget_kb=3000.0, seed=42, n_jobs=2, log=print):
        self.store, self.y, self.tr, self.va = store, y, tr, va
        self.lam_lat, self.lam_size = lam_lat, lam_size
        self.lat_budget, self.size_budget = lat_budget_ms, size_budget_kb
        self.seed, self.n_jobs, self.log = seed, n_jobs, log
        self.cache: dict[str, dict] = {}
        self.trace: list[dict] = []

    # ------------------------------------------------------------------ cost model
    def penalty(self, lat_ms: float, size_kb: float) -> float:
        return self.lam_lat * lat_ms / self.lat_budget + self.lam_size * size_kb / self.size_budget

    def measure_latency(self, pipe, cfg: FeatureConfig, idx: np.ndarray, n: int = 20) -> float:
        """Median per-image latency in ms: feature extraction + predict, one image at a time."""
        sel = idx[:n]
        imgs = self.store.images
        if "clf" in pipe.named_steps and hasattr(pipe.named_steps["clf"], "n_jobs"):
            pipe.named_steps["clf"].set_params(n_jobs=1)
        # warm-up (first call pays one-off import / allocation costs)
        pipe.predict(extract_one(imgs[sel[0]], cfg)[None])
        times = []
        for i in sel:
            t = time.perf_counter()
            f = extract_one(imgs[i], cfg)
            pipe.predict(f[None])
            times.append((time.perf_counter() - t) * 1000.0)
        return float(np.median(times))

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, params: dict) -> dict:
        params = canonical(params)
        key = repr(sorted(params.items()))
        if key in self.cache:
            return self.cache[key]
        cfg = feature_config(params)
        Xtr = self.store.matrix(cfg, self.tr)
        Xva = self.store.matrix(cfg, self.va)
        pipe = build_pipeline(params, Xtr.shape[1], probability=False, seed=self.seed, n_jobs=self.n_jobs)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, self.y[self.tr])
            pred = pipe.predict(Xva)
        fit_s = time.time() - t0
        acc = accuracy_score(self.y[self.va], pred)
        f1 = f1_score(self.y[self.va], pred, average="macro")
        size_kb = len(pickle.dumps(pipe)) / 1024.0
        lat = self.measure_latency(pipe, cfg, self.va)
        res = {"params": params, "val_acc": float(acc), "val_f1": float(f1), "latency_ms": lat,
               "size_kb": float(size_kb), "n_features": int(Xtr.shape[1]), "fit_seconds": fit_s,
               "fitness": float(acc - self.penalty(lat, size_kb))}
        self.cache[key] = res
        self.trace.append(res)
        self.log(f"  [eval {len(self.trace):3d}] {params['classifier']:3s} img={params['img_size']:3d} "
                 f"feat={res['n_features']:4d} acc={acc:.4f} lat={lat:5.1f}ms size={size_kb:7.0f}KB "
                 f"fit={res['fitness']:.4f} ({fit_s:.1f}s)")
        return res
