"""Clean the dataset for HarvestGuard training.
 
    python prepare_dataset.py
 
Reads the folders inside  dataset/   (any nesting, original names)
and writes              dataset_clean/<StandardClassName>/*.jpg  with
  - class folders renamed to the names the app expects,
  - nested folders flattened,
  - exact duplicate images removed.
Your original dataset/ folder is never changed.
Then train with:  python -m ml.train --data-dir dataset_clean --max-per-class 700 --pop 10 --iters 12
"""
import hashlib
import shutil
import sys
from pathlib import Path
 
SRC = Path("dataset")
OUT = Path("dataset_clean")
 
M = {
    'Pepper__bell___Bacterial_spot': 'Pepper,_bell___Bacterial_spot',
    'Pepper__bell___healthy': 'Pepper,_bell___healthy',
    'Potato___Early_blight': 'Potato___Early_blight',
    'Potato___Late_blight': 'Potato___Late_blight',
    'Potato___healthy': 'Potato___healthy',
    'Tomato_Bacterial_spot': 'Tomato___Bacterial_spot',
    'Tomato_Early_blight': 'Tomato___Early_blight',
    'Tomato_Late_blight': 'Tomato___Late_blight',
    'Tomato_Leaf_Mold': 'Tomato___Leaf_Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato___Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato___Spider_mitesTwo-spotted_spider_mite',
    'Tomato__Target_Spot': 'Tomato___Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato_healthy': 'Tomato___healthy',
}
KEYS = sorted(M, key=len, reverse=True)
 
 
def standard_name(folder_name):
    for k in KEYS:
        if folder_name.startswith(k):
            return M[k]
    # already-standard names (e.g. Tomato___Late_blight) are kept as they are
    if folder_name in M.values():
        return folder_name
    return None
 
 
def main():
    if not SRC.is_dir():
        sys.exit(f"Folder '{SRC}' not found. Put your dataset folders inside a folder called 'dataset' next to this file.")
    seen, counts, skipped = set(), {}, set()
    for f in SRC.rglob('*'):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        cls = next((standard_name(p.name) for p in [f.parent, *f.parents] if p != SRC.parent and standard_name(p.name)), None)
        if not cls:
            skipped.add(f.parent.name)
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        (OUT / cls).mkdir(parents=True, exist_ok=True)
        shutil.copy(f, OUT / cls / f"{h[:10]}_{f.name}")
        counts[cls] = counts.get(cls, 0) + 1
    for k in sorted(counts):
        print(f"{k:48s} {counts[k]:5d}")
    print(f"\ncopied {sum(counts.values())} unique images, {len(counts)} classes -> {OUT}")
    if skipped:
        print("WARNING: images in these folders were skipped (name not recognised):", sorted(skipped))
    if len(counts) != 15:
        print("WARNING: expected 15 classes. Check your dataset folder before training.")
 
 
if __name__ == "__main__":
    main()
 