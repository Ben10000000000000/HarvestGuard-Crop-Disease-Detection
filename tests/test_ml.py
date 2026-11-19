import numpy as np
from ml.features import FeatureConfig, block_keys, compute_block, extract_one
from ml.search_space import DIM, build_pipeline, canonical, decode, encode_default, feature_config
from ml.woa import woa_maximize


def test_woa_finds_optimum_of_sphere():
    target = np.array([0.2, 0.7, 0.4, 0.9, 0.1])
    res = woa_maximize(lambda x: -float(((x - target) ** 2).sum()), 5, pop_size=14, iters=40, seed=3)
    assert res["best_fit"] > -0.02
    assert res["history"][-1]["best"] >= res["history"][0]["best"]          # elitist: never gets worse
    assert res["n_calls"] == 14 * 41


def test_woa_is_reproducible_and_bounded():
    f = lambda x: float(x.sum())
    a = woa_maximize(f, 4, 8, 10, seed=1); b = woa_maximize(f, 4, 8, 10, seed=1)
    assert np.allclose(a["best_pos"], b["best_pos"]) and ((a["best_pos"] >= 0) & (a["best_pos"] <= 1)).all()


def test_decode_covers_space_and_default_is_valid():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = decode(rng.random(DIM))
        assert p["img_size"] in (32, 48, 64, 96) and p["classifier"] in ("svm", "mlp", "rf")
    assert decode(np.zeros(DIM))["img_size"] == 32 and decode(np.ones(DIM))["img_size"] == 96
    assert decode(encode_default())["classifier"] == "svm"
    assert canonical(dict(decode(np.zeros(DIM)), use_lbp=False, lbp_radius=3))["lbp_radius"] == 1


def test_feature_vector_matches_block_concat_and_is_finite():
    img = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    cfg = FeatureConfig(48, 8, True, 2, True, 8, True, True)
    v = extract_one(img, cfg)
    assert np.allclose(v, np.concatenate([compute_block(img, k) for k in block_keys(cfg)]))
    assert np.isfinite(v).all() and v.dtype == np.float32


def test_every_classifier_builds_and_fits():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 20)), np.repeat([0, 1, 2], 100)
    X[:, :6] += y[:, None] * 3
    for clf, extra in (("svm", {"C": 1, "gamma_mult": 1}), ("mlp", {"hidden": 16, "alpha": 1e-3}),
                       ("rf", {"n_estimators": 20, "max_depth": 8})):
        pipe = build_pipeline({"classifier": clf, "pca": 16, **extra}, 20, probability=True).fit(X, y)
        assert pipe.score(X, y) > 0.9 and pipe.predict_proba(X[:2]).shape == (2, 3)
