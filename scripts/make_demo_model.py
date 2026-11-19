"""Builds a tiny throw-away model on synthetic colour blobs (ONLY for UI smoke tests, not for real use)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import numpy as np

from ml.features import FeatureConfig, extract_one
from ml.search_space import build_pipeline

out = sys.argv[1]
rng = np.random.default_rng(0)
classes = ["Tomato___healthy", "Tomato___Late_blight", "Potato___Early_blight"]
cfg = FeatureConfig(32, 6, True, 1, False)
X, y = [], []
for c, base in enumerate([[40, 160, 60], [90, 70, 40], [150, 130, 50]]):
    for _ in range(40):
        X.append(extract_one(np.clip(np.array(base) + rng.normal(0, 18, (128, 128, 3)), 0, 255).astype(np.uint8), cfg)); y.append(c)
params = {"classifier": "svm", "C": 5.0, "gamma_mult": 1.0, "pca": 0}
pipe = build_pipeline(params, len(X[0]), probability=True).fit(np.array(X), np.array(y))
joblib.dump({"pipeline": pipe, "classes": classes, "feature_config": cfg.to_dict(), "params": params,
             "meta": {"test_acc": 0.0, "latency_ms": 1.0, "size_kb": 10.0, "test_macro_f1": 0.0}}, out)
