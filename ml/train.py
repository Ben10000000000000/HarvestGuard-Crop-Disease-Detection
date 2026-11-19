"""End-to-end training: baseline vs random search vs Whale Optimization Algorithm.

    python -m ml.train --data-dir /path/to/plantvillage --max-per-class 150 --pop 12 --iters 15

Outputs (default): models/harvestguard_model.joblib, reports/*.json|png|csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .dataset import load_dataset, split
from .features import extract_one
from .objective import Evaluator, FeatureStore
from .search_space import DIM, build_pipeline, canonical, decode, encode_default, feature_config
from .woa import woa_maximize
from . import report


def final_fit(params, store, y, fit_idx, test_idx, seed, evaluator):
    """Refit on train+val with probability outputs, score on the untouched test split."""
    cfg = feature_config(params)
    Xfit, Xte = store.matrix(cfg, fit_idx), store.matrix(cfg, test_idx)
    pipe = build_pipeline(params, Xfit.shape[1], probability=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(Xfit, y[fit_idx])
        pred = pipe.predict(Xte)
    lat = evaluator.measure_latency(pipe, cfg, test_idx, n=40)
    return pipe, cfg, pred, {
        "test_acc": float(accuracy_score(y[test_idx], pred)),
        "test_macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
        "latency_ms": lat,
        "size_kb": len(pickle.dumps(pipe)) / 1024.0,
        "n_features": int(Xfit.shape[1]),
        "params": params,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam-lat", type=float, default=0.02)
    ap.add_argument("--lam-size", type=float, default=0.02)
    ap.add_argument("--lat-budget-ms", type=float, default=25.0)
    ap.add_argument("--size-budget-kb", type=float, default=3000.0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--model-out", default="models/harvestguard_model.joblib")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--cache-dir", default="data")
    a = ap.parse_args()

    t_start = time.time()
    Path(a.report_dir).mkdir(exist_ok=True, parents=True)
    Path(a.model_out).parent.mkdir(exist_ok=True, parents=True)

    X, y, classes = load_dataset(a.data_dir, a.max_per_class, a.cache_dir, a.n_jobs, seed=a.seed)
    tr, va, te = split(X, y, seed=a.seed)
    print(f"[data] {len(y)} images, {len(classes)} classes -> train {len(tr)} / val {len(va)} / test {len(te)}")
    store = FeatureStore(X, n_jobs=a.n_jobs)
    ev = Evaluator(store, y, tr, va, a.lam_lat, a.lam_size, a.lat_budget_ms, a.size_budget_kb,
                   seed=a.seed, n_jobs=a.n_jobs)
    fit_idx = np.concatenate([tr, va])
    results = {}

    # 1) hand-tuned baseline ------------------------------------------------------------
    print("\n== Baseline (hand-tuned SVM) ==")
    base_params = canonical(decode(encode_default()))
    ev.evaluate(base_params)
    results["baseline"] = {"params": base_params}

    # 2) Whale Optimization Algorithm ---------------------------------------------------
    print(f"\n== Whale Optimization Algorithm (pop={a.pop}, iters={a.iters}) ==")
    woa_hist = []

    def fobj(g):
        return ev.evaluate(decode(g))["fitness"]

    woa = woa_maximize(fobj, DIM, pop_size=a.pop, iters=a.iters, seed=a.seed,
                       on_iter=lambda h: (woa_hist.append(h),
                                          print(f"[WOA] iter {h['iter']:2d}  best fitness {h['best']:.4f}  "
                                                f"mean {h['mean']:.4f}  unique evals {len(ev.cache)}", flush=True)))
    woa_params = canonical(decode(woa["best_pos"]))
    n_woa_calls = woa["n_calls"]

    # 3) Random search with the same number of fitness calls ---------------------------
    print(f"\n== Random search ({n_woa_calls} samples, same budget) ==")
    rng = np.random.default_rng(a.seed + 1)
    rs_best, rs_hist = None, []
    for i in range(n_woa_calls):
        r = ev.evaluate(decode(rng.random(DIM)))
        if rs_best is None or r["fitness"] > rs_best["fitness"]:
            rs_best = r
        rs_hist.append(rs_best["fitness"])
    rs_params = rs_best["params"]

    # 4) Final comparison on the untouched test split ----------------------------------
    print("\n== Final refit on train+val, evaluation on held-out test split ==")
    finals, models = {}, {}
    for name, p in (("baseline", base_params), ("random_search", rs_params), ("woa", woa_params)):
        pipe, cfg, pred, res = final_fit(p, store, y, fit_idx, te, a.seed, ev)
        res["val_fitness"] = ev.evaluate(p)["fitness"]
        res["val_acc"] = ev.evaluate(p)["val_acc"]
        finals[name], models[name] = res, (pipe, cfg, pred)
        print(f"  {name:14s} test acc {res['test_acc']:.4f}  macro-F1 {res['test_macro_f1']:.4f}  "
              f"latency {res['latency_ms']:.1f} ms  size {res['size_kb']:.0f} KB  ({p['classifier']})")

    pipe, cfg, pred = models["woa"]
    cm = confusion_matrix(y[te], pred, labels=range(len(classes)))
    rep = classification_report(y[te], pred, labels=range(len(classes)), target_names=classes,
                                output_dict=True, zero_division=0)

    meta = {
        "dataset": "PlantVillage", "n_images": int(len(y)), "n_classes": len(classes),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "woa": {"pop": a.pop, "iters": a.iters, "fitness_calls": n_woa_calls,
                "unique_configs_trained": len(ev.cache), "seconds": woa["seconds"]},
        "fitness": {"formula": "val_acc - lam_lat*latency/budget_lat - lam_size*size/budget_size",
                    "lam_lat": a.lam_lat, "lam_size": a.lam_size,
                    "latency_budget_ms": a.lat_budget_ms, "size_budget_kb": a.size_budget_kb},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_minutes": round((time.time() - t_start) / 60, 1),
        "test_acc": finals["woa"]["test_acc"], "test_macro_f1": finals["woa"]["test_macro_f1"],
        "latency_ms": finals["woa"]["latency_ms"], "size_kb": finals["woa"]["size_kb"],
    }
    joblib.dump({"pipeline": pipe, "classes": classes, "feature_config": cfg.to_dict(),
                 "params": woa_params, "meta": meta}, a.model_out, compress=3)

    out = {"meta": meta, "comparison": finals, "woa_history": woa_hist,
           "random_search_history": rs_hist, "woa_best_params": woa_params,
           "woa_top_configs": sorted(ev.trace, key=lambda r: -r["fitness"])[:10],
           "all_evaluations": ev.trace}
    Path(a.report_dir, "results.json").write_text(json.dumps(out, indent=2, default=float))
    Path(a.report_dir, "classification_report.json").write_text(json.dumps(rep, indent=2))
    report.make_plots(a.report_dir, woa_hist, rs_hist, finals, cm, classes, ev.trace)
    print(f"\nSaved model -> {a.model_out}\nTotal time {meta['total_minutes']} min")


if __name__ == "__main__":
    main()
