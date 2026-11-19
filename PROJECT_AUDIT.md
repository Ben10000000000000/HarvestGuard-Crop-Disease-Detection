# HarvestGuard project audit (before adding your own dataset)

## Alignment with the requested project
- **Goal:** leaf crop-disease image classification with low-compute inference.
- **WOA:** `ml/woa.py` searches a configuration space defined in `ml/search_space.py`; `ml/objective.py` scores validation accuracy with latency and model-size penalties.
- **Dataset loader:** `ml/dataset.py` supports class-folder datasets, including PlantVillage's `color/<class>/<image>` layout.
- **Evaluation:** `ml/train.py` compares a baseline, random search, and WOA with stratified train/validation/test partitions. The test split is intended to be held out until final evaluation.
- **Web app:** Flask pages provide sign-up, sign-in, password reset, image detection, private scan history, and a model report.
- **UI update:** home page now has a lightweight animated depth/3D-style background and floating illustration; sign-in, sign-up, forgot-password, and reset-password cards use matching animated styling. CSS respects reduced-motion preferences and does not require WebGL.

## Important limitations to understand
1. The included model was trained on a **5,700-image subset** (150 images per class), not your not-yet-uploaded dataset and not the full PlantVillage dataset. Included report metrics are from that subset's held-out test split: about **88.3% accuracy** and **88.2% macro-F1**. These are existing report values, not a guarantee of field accuracy.
2. The current model bundle was created with scikit-learn **1.9.1**. Use that version when loading the included pickle/joblib model to avoid cross-version warnings or incompatibility.
3. PlantVillage images often have clean backgrounds; real field images can perform worse. The app should be evaluated on genuinely separate field photos before claiming field-ready accuracy.
4. This package has not been retrained on your own dataset because you have not uploaded it yet. When you provide it, inspect the folder structure and labels first, then train and evaluate without mixing duplicate/near-duplicate images across splits.

## Recommended training flow after you provide the dataset
1. Keep the original dataset untouched and inspect class counts, unreadable images, duplicates, and label-folder names.
2. Split data by image/source (or plant/session where known) to reduce leakage.
3. Run the same baseline/random-search/WOA comparison with a fixed seed.
4. Review per-class precision/recall/F1, confusion matrix, test accuracy, CPU latency, and model size.
5. Save the final model and metrics together, and verify that the web app loads that exact model.

## Verification status
The source files were inspected and the bundled model/report metadata were readable in this environment. Automated pytest tests could not be run here because Flask is not installed in the current execution environment. Run `pip install -r requirements.txt` in your project environment, then `python -m pytest -q tests`.
