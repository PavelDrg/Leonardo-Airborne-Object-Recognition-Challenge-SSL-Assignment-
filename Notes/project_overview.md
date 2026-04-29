# Leonardo Airborne Object Recognition

## Goal
Build an offline object detection pipeline for a small-object, imbalanced aerial dataset.

## Data handling choice
- Keep the original dataset unchanged.
- Create a processed mirror with:
  - image-level train/val split
  - YOLO labels
  - manifests for reproducibility
  - summary JSON for before/after comparison

## Why this approach
- The task is detection, so box-level leakage must be avoided.
- The objects are tiny, so training-time resizing is preferable to permanently shrinking the dataset.
- A processed mirror allows easy experiments without destroying the raw data.

## Main risks
- class imbalance
- tiny boxes
- very dense images
- malformed coordinates in the CSV

## Practical experiments
1. Baseline: small YOLO model, moderate resolution.
2. Improved: larger YOLO model, higher resolution, stronger augmentation.
3. Optional: tiled inference or SAHI if small objects remain hard.

## Outputs
- `processed/` dataset mirror
- `Notes/eda/` plots and JSON summaries
- model training logs and later a short report
