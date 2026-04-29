# Project Notes

## Task
Offline object detection on the Leonardo airborne object dataset.

Goal:
- understand the dataset
- preprocess it safely
- train a baseline detector
- train an improved detector
- write a short report and presentation

## Main data source
- Kaggle dataset mirror: `leonard-data`
- Local raw dataset folder: `leonardo-airborne-object-recognition-challenge/`

## What is in this repo

### `scripts/`
- `preprocess_dataset.py` - builds the processed YOLO mirror, split, and manifests
- `eda_dataset.py` - generates raw and processed dataset plots
- `train_yolo.py` - trains the baseline or improved YOLO model
- `evaluate_yolo.py` - standalone evaluation if needed

### `processed/`
- YOLO-style training data
- `data.yaml`
- `images/train`, `images/val`
- `labels/train`, `labels/val`
- `meta/summary.json`
- `meta/train_manifest.csv`
- `meta/val_manifest.csv`

### `Notes/`
- `project_overview.md` - short architecture summary
- `run_order.md` - exact command order
- `baseline_observations.md` - baseline + improved findings
- `improved_run_plan.md` - why the improved run was changed
- `eda_raw/` - raw-data plots and summaries
- `eda_processed/` and `eda_processed_v2/` - post-preprocessing comparison plots

### `Report/`
- `main.tex` - report draft
- `references.bib` - bibliography
- `images/` - figures used in the report

### `Prezentare/`
- Beamer presentation template
- `main.tex` - 6-slide presentation draft
- `images/` - figures used in the slides

## Workflow that was used
1. Inspect raw dataset statistics.
2. Preprocess to YOLO format.
3. Re-run EDA on the processed split.
4. Train baseline YOLO11n model.
5. Evaluate baseline and inspect plots/images.
6. Train improved YOLO11s model with rare-class oversampling.
7. Use the improved results in the report/presentation.

## Important findings
- The task is small-object detection, not classification.
- The dataset is imbalanced.
- Boxes are often tiny.
- Some boxes needed clipping and a few were invalid.
- Image-level splitting is required to avoid leakage.

## Baseline results
- precision: `0.790`
- recall: `0.642`
- mAP50: `0.715`
- mAP50-95: `0.382`

## Improved results
- precision: `0.834`
- recall: `0.716`
- mAP50: `0.792`
- mAP50-95: `0.439`

## Training setup
- Baseline: `yolo11n`, `960` px, `18` epochs
- Improved: `yolo11s`, `1280` px, `20` epochs, rare-class oversampling
- Both runs used CUDA and Ultralytics YOLO

## Best model files
- Baseline: `runs/detect/runs/detect/leonardo_baseline/weights/best.pt`
- Improved: `runs/detect/leonardo_improved-4/weights/best.pt`

## Useful commands
```bash
python scripts/eda_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --output-dir "Notes/eda_raw"
python scripts/preprocess_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --output-root "processed" --force
python scripts/eda_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --processed-root "processed" --output-dir "Notes/eda_processed_v2"
python scripts/train_yolo.py --data "processed/data.yaml" --variant baseline --project "runs/detect"
python scripts/train_yolo.py --data "processed/data.yaml" --variant improved --oversample-rare --oversample-factor 2 --project "runs/detect"
```

## If you need to explain the pipeline fast
- raw annotations come from `train.csv`
- preprocessing converts them to YOLO labels
- YOLO learns bounding boxes and class labels
- evaluation uses standard detection metrics and plots
