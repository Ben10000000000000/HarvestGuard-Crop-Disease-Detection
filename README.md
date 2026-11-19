# 🌿 HarvestGuard — Whale-Optimization-tuned crop-disease detection

HarvestGuard classifies a photo of a leaf into one of **15 crop–disease classes of the PlantVillage dataset** (bell pepper, potato, tomato) and gives practical management advice.
The classifier is **tuned by the Whale Optimization Algorithm (WOA)** to maximise accuracy *under a compute budget*, so it can run on low-power field devices (Raspberry Pi, old laptops, phones behind a small server) without a GPU.

The project has two parts:

| Part | What it does | Where |
|---|---|---|
| **ML pipeline** | Feature extraction → WOA search over features + classifier + hyper-parameters → evaluation vs. baselines | `ml/` |
| **Web app** | Sign-up with e-mail OTP verification / login / forgot-password by OTP, leaf upload, diagnosis + advice, private scan history, optimisation report | `app/`, `run.py` |

---

## 0. Current trained model (trained on YOUR dataset)

Trained on **9,625 unique PlantVillage images, 15 classes** (capped at 700/class from the 19,418 unique images you supplied; exact duplicates across your zip files were removed so none can leak between train and test). Stratified 70/15/15 split, seed 42, WOA population 10 × 12 iterations (27 min on 2 CPU cores).

| Method (same fitness-evaluation budget) | Test accuracy | Macro-F1 | Latency / image | Model size |
|---|---|---|---|---|
| Hand-tuned baseline (SVM) | 89.6 % | 0.891 | 7.8 ms | 18.6 MB |
| Random search | 93.2 % | 0.930 | 4.9 ms | 148 KB |
| **Whale Optimization (shipped)** | **95.0 %** | **0.944** | 9.8 ms | **242 KB** |

Test split = 1,444 images never used during the search. Weakest class: Potato healthy (F1 0.83, only 152 images exist, so 23 test images); see `reports/classification_report.json` and `reports/confusion_matrix.png`. Accuracy is on clean, lab-style PlantVillage photos; expect lower accuracy on cluttered field photos. The images in `samples/` come from the training pool, so use them to try the app, not to measure accuracy.

To retrain on **all 19,418 images** (slower, a few hours on 2 cores): `python -m ml.train --data-dir <folder-with-one-subfolder-per-class> --max-per-class 0 --pop 10 --iters 12`. Folder names must match the class names used by `app/advisor.py` (e.g. `Tomato___Late_blight`).

## 0b. 3D website
Every page now has a real-time WebGL2 3D crop field (`app/static/js/bg3d.js`: ~50k wind-animated blades, sun, clouds, fog, pollen, mouse-parallax) behind frosted-glass panels. It falls back to a plain gradient if WebGL is unavailable and honours reduced-motion. No external libraries or CDN (compatible with the app's strict CSP).

## 0c. Private scan history
`/history` lists **only the signed-in user's** scans (every MongoDB query filters on `user_id`); opening, viewing the image of, or deleting another user's scan returns 404. Covered by `tests/test_history_isolation.py` (run `python -m pytest -q tests`).

---

## 1. Quick start

```bash
cd harvestguard
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                                           # → http://127.0.0.1:5000
```

A trained model (`models/harvestguard_model.joblib`) and its report (`reports/`) are already included, so the app works immediately.
Create an account, open **Scan a leaf** and try the images in `samples/`.

### Re-train on the full Kaggle PlantVillage dataset (recommended)

The bundled model was trained on a **150-images-per-class subset (5,700 images)** so it could be built quickly. For your submission, train on the full dataset:

```bash
pip install kaggle                       # put your kaggle.json in ~/.kaggle/
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/ --unzip
python -m ml.train --data-dir data/plantvillage_dataset --max-per-class 0 --pop 12 --iters 15
```

* `--data-dir` can be any folder that contains `<ClassName>/<image>.jpg` (the loader automatically finds the `color/` folder of the Kaggle layout).
* `--max-per-class 0` uses **all** images (~54k). With a cap such as `--max-per-class 400` training is faster.
* Useful flags: `--pop` (whales), `--iters`, `--seed`, `--lat-budget-ms`, `--size-budget-kb`, `--lam-lat`, `--lam-size`, `--n-jobs`.
* Outputs: `models/harvestguard_model.joblib`, `reports/results.json`, `reports/*.png`, `reports/classification_report.json`.
  Restart `python run.py` afterwards; the **Model** page in the app shows the new report.

---

## 2. How the Whale Optimization Algorithm is used

**Goal:** *maximise accuracy on low-compute devices.* Accuracy alone would favour huge models, so every candidate is scored with a **cost-aware fitness**:

```
fitness = val_accuracy − λ_lat · (latency_ms / latency_budget) − λ_size · (model_KB / size_budget)
```

* **latency** is measured for real: feature extraction + prediction for **one image on one CPU thread** (median of 20–40 images).
* **size** is the pickled pipeline size.
* defaults: budgets 25 ms and 3 MB, λ = 0.02 each (see `ml/objective.py`).

**A whale = 12 genes in [0, 1]** (`ml/search_space.py`) that decode into a full configuration:

| Genes | Decoded to |
|---|---|
| `img_size` | 32 / 48 / 64 / 96 px working resolution |
| `hsv_bins` | colour-histogram resolution (4 … 16 bins) |
| `use_lbp`, `lbp_radius` | Local-Binary-Pattern texture block and its radius |
| `use_hog`, `hog_ppc` | HOG shape block and cell size |
| `use_jhist`, `use_glcm` | joint HSV histogram, GLCM texture block |
| `pca` | none / 16 / 32 / 64 / 96 components |
| `classifier` | SVM-RBF · MLP · Random Forest |
| `hp1`, `hp2` | `C`, `γ` (SVM) · hidden units, L2 (MLP) · trees, depth (RF) |

**The algorithm** (`ml/woa.py`, Mirjalili & Lewis 2016): every iteration each whale either *encircles* the best solution (|A|<1), *searches* around a random whale (|A|≥1, exploration) or follows the *bubble-net spiral* around the best (p ≥ 0.5). The coefficient `a` shrinks 2 → 0 so the swarm moves from exploration to exploitation. Duplicate configurations are cached, so repeated whales cost nothing.

**Fair comparison** (all in `ml/train.py`): a hand-tuned baseline, **random search with exactly the same number of fitness evaluations**, and WOA are all refit on train+validation and scored **once** on the same untouched test split (70 / 15 / 15 stratified). The results table and plots are written to `reports/` and shown on the app's **Model** page.

Standard WOA is run with population 12 × 15 iterations by default. Increase `--pop`/`--iters` for a more thorough search.

---

## 3. The web app

* **Authentication** (`app/auth.py`, `app/security.py`)
  * Sign-up with name, e-mail, password + confirmation, live strength meter; policy: ≥ 8 chars, letter + number, not a common password, not containing your e-mail name.
  * Passwords stored only as salted **scrypt** hashes (Werkzeug).
  * Login with generic error messages (no user enumeration), **account lock-out** after 5 failures (10 min), per-IP rate limits, session rotation on login / password change (anti-fixation), "keep me signed in".
  * **CSRF tokens** on every POST, safe `next=` redirects, HttpOnly + SameSite cookies, strict **Content-Security-Policy** (no inline scripts/styles), `X-Frame-Options: DENY`.
  * **Email verification:** sign-up sends a 6-digit OTP; the account can't sign in until it is confirmed.
  * **Forgot password:** a 6-digit OTP is e-mailed, then the user chooses a new password. Codes are single-use, expire in 10 minutes, are stored only as an HMAC hash, burn after 5 wrong tries, and can be re-sent once every 60 s.
  * Account page: edit profile, change password, **delete account and all data**.
  * Each user can only see their own scans and photos (image URLs are access-checked).
* **Diagnosis**: upload or capture (mobile camera) a leaf → top-3 predictions with confidence, a *low-confidence* warning, and crop-specific management steps (`app/advisor.py`). Uploads are validated, re-encoded to JPEG (metadata stripped) and stored per user.
* **History / dashboard**: filter healthy vs. diseased, counts, most frequent problems, delete scans.
* **JSON API**: `POST /api/predict` (multipart field `leaf`, same login session + `X-CSRF-Token` header).
* **Storage**: SQLite file in `instance/` — no database server to install.

### OTP e-mails
If no SMTP server is configured the OTP is **printed in the terminal** where `python run.py` runs. To send real e-mails copy `.env.example` → `.env` and fill the `HG_SMTP_*` values (Gmail needs an *App Password*). For a **local demo only** set `HG_SHOW_OTP=1` to show the code on the page — never enable that on a public server. Accounts created before OTP existed are treated as already verified.

### Deploying
`python run.py` uses **waitress** when installed (it is in `requirements.txt`). Put it behind HTTPS (nginx/Caddy) and set `HG_COOKIE_SECURE=1`. Set a fixed `SECRET_KEY` in `.env` if you run several workers.

---

## 4. Project layout

```
harvestguard/
├── run.py                  start the web app
├── requirements.txt  .env.example
├── ml/
│   ├── features.py         colour / LBP / HOG / joint-HSV / GLCM feature blocks
│   ├── woa.py              Whale Optimization Algorithm (maximisation)
│   ├── search_space.py     gene vector → configuration → scikit-learn pipeline
│   ├── objective.py        cost-aware fitness (accuracy − latency − size penalty)
│   ├── dataset.py          PlantVillage loader, cache, stratified split
│   ├── train.py            baseline vs random search vs WOA, final test evaluation
│   └── report.py           convergence / comparison / confusion-matrix plots
├── app/                    Flask app (auth, views, inference, advisor, templates, static)
├── models/                 harvestguard_model.joblib (trained WOA model)
├── reports/                results.json, PNG plots, classification report, training log
├── samples/                8 example leaf photos
├── scripts/make_demo_model.py   tiny synthetic model for UI smoke tests only
└── tests/                  pytest suite (auth, CSRF, lock-out, reset flow, detection, WOA, features)
```

Run the tests with `python -m pytest -q tests`.

---

## 5. Honest limitations

* **PlantVillage is a laboratory dataset** (single leaf, uniform background). Accuracy on cluttered *real field* photos will be noticeably lower — report test accuracy as "on PlantVillage test split", and ideally fine-tune on photos from your own fields.
* The model uses hand-crafted features + classical classifiers, chosen deliberately because they run in milliseconds without a GPU. A fine-tuned MobileNet would score higher on accuracy but needs a heavier runtime; the WOA framework here can be extended to tune such a network (image size, width multiplier, learning rate, quantisation) by replacing `ml/objective.py`.
* The advice texts are general integrated-pest-management guidance, **not** a substitute for local extension services; chemical products and rates depend on country and label.
* Plants outside the 15 classes (and non-leaf images) can still receive a prediction; the low-confidence warning helps but is not a guarantee.
* Dataset: Hughes & Salathé (2015), *An open access repository of images on plant health…*, arXiv:1511.08060; Mohanty et al. (2016), *Using deep learning for image-based plant disease detection*. WOA: Mirjalili & Lewis (2016), *The Whale Optimization Algorithm*, Advances in Engineering Software 95.

## MongoDB configuration (current build)

Normal app runs use MongoDB (not SQLite) for user accounts, password-reset tokens,
audit events, and per-user prediction history. The default local connection is:

```env
HG_MONGO_URI=mongodb://localhost:27017
HG_MONGO_DB=harvestguard_db
```

Copy `.env.example` to `.env` and change `HG_MONGO_URI` if using MongoDB Atlas or a
non-default local connection. Keep credentials out of source control. MongoDB must
be running and reachable before starting the app. The app creates unique/indexed
collections on startup. SQLite is retained only as a temporary database backend for
the automated tests when a test explicitly supplies `DATABASE`.

Install dependencies with `python -m pip install -r requirements.txt`, then start
with `python run.py` (or the launch command documented above). If MongoDB is offline,
the app intentionally reports a connection error rather than silently storing data
in a different database.
