"""Whale Optimization Algorithm (Mirjalili & Lewis, 2016) - maximisation version.

Each whale is a point in [0, 1]^dim. Three behaviours, chosen per whale and iteration:
  * encircling prey      (|A| < 1, p < 0.5): move towards the best whale
  * searching for prey   (|A| >= 1, p < 0.5): move relative to a random whale (exploration)
  * bubble-net spiral    (p >= 0.5): logarithmic spiral around the best whale
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np


def woa_maximize(
    fitness: Callable[[np.ndarray], float],
    dim: int,
    pop_size: int = 12,
    iters: int = 15,
    seed: int = 0,
    b: float = 1.0,
    init: Optional[np.ndarray] = None,
    on_iter: Optional[Callable[[dict], None]] = None,
) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.random((pop_size, dim))
    if init is not None:  # optional warm-start whales (rows in [0,1])
        init = np.clip(np.atleast_2d(init), 0, 1)
        X[: len(init)] = init[:pop_size]

    fit = np.full(pop_size, -np.inf)
    best_pos, best_fit = X[0].copy(), -np.inf
    history, n_calls = [], 0
    t0 = time.time()

    def evaluate(i):
        nonlocal n_calls
        n_calls += 1
        return float(fitness(X[i]))

    for i in range(pop_size):
        fit[i] = evaluate(i)
        if fit[i] > best_fit:
            best_fit, best_pos = fit[i], X[i].copy()
    history.append({"iter": 0, "best": best_fit, "mean": float(fit.mean()), "calls": n_calls})
    if on_iter:
        on_iter(history[-1])

    for t in range(1, iters + 1):
        a = 2.0 - 2.0 * t / iters            # linearly 2 -> 0
        for i in range(pop_size):
            r1, r2 = rng.random(dim), rng.random(dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            p, l = rng.random(), rng.uniform(-1, 1)
            if p < 0.5:
                if np.abs(A).mean() < 1:      # exploitation: encircle best
                    D = np.abs(C * best_pos - X[i])
                    X[i] = best_pos - A * D
                else:                         # exploration: follow a random whale
                    rand = X[rng.integers(pop_size)]
                    D = np.abs(C * rand - X[i])
                    X[i] = rand - A * D
            else:                             # bubble-net spiral
                D = np.abs(best_pos - X[i])
                X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
            X[i] = np.clip(X[i], 0.0, 1.0)
            fit[i] = evaluate(i)
            if fit[i] > best_fit:
                best_fit, best_pos = fit[i], X[i].copy()
        history.append({"iter": t, "best": best_fit, "mean": float(fit.mean()), "calls": n_calls})
        if on_iter:
            on_iter(history[-1])

    return {"best_pos": best_pos, "best_fit": best_fit, "history": history,
            "n_calls": n_calls, "seconds": time.time() - t0}
