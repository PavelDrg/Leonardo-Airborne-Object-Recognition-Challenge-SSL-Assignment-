# Final Improvement Plan

## Teacher Notes Assessment

The teacher's notes are useful, but they need to be translated into concrete experiments for our current pipeline.

| Teacher note | Validity | How it applies here |
|---|---|---|
| "It is a bounding box segmentation and classification problem" | Partly valid | It is definitely localization plus classification, but with the current dataset it is object detection, not true mask segmentation. We only have bounding boxes, not pixel masks. Calling it segmentation in the report would be inaccurate unless we generate or receive masks. |
| "Train different YOLOs, one for larger objects, one for smaller objects / per class" | Valid as an advanced idea | A class-specific or scale-specific ensemble could help, but it is expensive and more complex. It should come after simpler high-impact work like slicing and threshold tuning. |
| "Focus loss" | Directionally valid | The likely intended idea is focal loss. This targets class imbalance and hard examples. It is relevant, but Ultralytics YOLO11 does not expose a simple focal-loss switch in the current training script, so this is a medium-risk modification. |
| "Cut up pictures so details are not lost" | Very valid | This is the best next direction. The dataset has many tiny boxes, and resizing full images to 960 or 1280 still loses detail. Sliced training/inference directly addresses the main failure mode. |

## Current State

The baseline and improved runs already show a clear improvement path:

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline `yolo11n`, 960 px | `0.790` | `0.642` | `0.715` | `0.382` |
| Improved `yolo11s`, 1280 px, oversampling | `0.836` | `0.715` | `0.792` | `0.439` |

The improved run helped because it used higher resolution, a larger model, stronger augmentation, rare-class oversampling, and a higher maximum detection count. The remaining failure mode is still mostly missed tiny objects and dense scenes. Therefore, the final part should not just train another generic model. It should specifically attack small-object detection.

## Best Direction

The strongest next step is sliced/tiled training and sliced inference.

Reasoning:
- Median box area is around `0.0015` of the image, so many targets are extremely small.
- Some images contain dense scenes, with object counts going above `100` and one image reaching `359` objects.
- Full-image resizing makes small boxes lose detail.
- Tiling keeps local resolution higher because each crop is resized independently.
- The teacher explicitly suggested this, and it matches the observed failure mode.

Expected result:
- Recall should improve more than precision.
- `Human`, `Obstacle`, and `GroundVehicle` should benefit because they are often small and dense.
- mAP50-95 should improve if tiled crops improve box localization.

## Plan Overview

Priority order:

1. Add sliced dataset generation for training and validation analysis.
2. Train a sliced-data YOLO model using the current improved settings as the starting point.
3. Add sliced inference/evaluation using SAHI-style merging or a custom tile-merging script.
4. Tune confidence and NMS thresholds on validation predictions.
5. Optionally train specialist models only if slicing gives a measurable improvement.
6. Only attempt focal loss if time remains, because it requires deeper training-code changes.

## Experiment 1: Sliced Dataset Training

### Goal
Create a new processed dataset where each original image is cut into overlapping tiles, and labels are transformed into tile-local YOLO coordinates.

### Proposed tile settings

| Parameter | Initial value | Reason |
|---|---:|---|
| Tile size | `640` or `768` px | Keeps small objects visible while staying GPU-friendly |
| Overlap | `20%` to `25%` | Prevents objects near tile borders from being cut too aggressively |
| Minimum retained box area | `20%` to `30%` of original box area | Avoids training on tiny fragments of objects |
| Keep empty tiles | limited sample | Useful negatives, but too many empty crops would dominate training |
| Splitting rule | keep original train/val split | Prevents leakage from tiles of the same image crossing splits |

### Script to add
Add a new script:

```text
scripts/preprocess_sliced_dataset.py
```

Responsibilities:
- Read `processed/meta/train_manifest.csv` and `processed/meta/val_manifest.csv` so the existing split is reused.
- For every image, create overlapping tiles.
- Convert original normalized boxes to pixel boxes.
- Intersect boxes with each tile.
- Drop boxes whose retained area is too small.
- Write tile images to `processed_sliced/images/train` and `processed_sliced/images/val`.
- Write YOLO labels to `processed_sliced/labels/train` and `processed_sliced/labels/val`.
- Write `processed_sliced/data.yaml`.
- Write `processed_sliced/meta/summary.json` with tile counts, object counts, empty tile counts, and per-class counts.

### Why this is better than only increasing `imgsz`
Increasing `imgsz` from 960 to 1280 already helped, but it is expensive and still resizes the whole scene. Tiling gives the model a zoomed-in view without requiring a massive full-image resolution.

### First training command
Use the improved model settings, but start with a smaller image size because the tiles are already zoomed-in:

```bash
python scripts/train_yolo.py --data "processed_sliced/data.yaml" --variant improved --project "runs/detect" --name "leonardo_sliced_yolo11s"
```

If GPU memory allows it, keep `imgsz=1280`; otherwise use `960` or adapt the improved config for sliced data.

### Success criteria
- Beat improved run recall: target above `0.715`.
- Beat improved run mAP50: target above `0.792`.
- Ideally beat improved run mAP50-95: target above `0.439`.
- Show qualitative improvement on dense validation images.

## Experiment 2: Sliced Inference on the Existing Improved Model

### Goal
Check whether the existing `leonardo_improved-4` model performs better if validation images are sliced at inference time.

This is faster than retraining and directly tests the teacher's tiling idea.

### Approach
Use the improved checkpoint:

```text
runs/detect/leonardo_improved-4/weights/best.pt
```

Run detection on overlapping tiles, convert tile predictions back to original image coordinates, and merge boxes using NMS or weighted boxes fusion.

### Script to add
Add a new script:

```text
scripts/evaluate_sliced_yolo.py
```

Responsibilities:
- Load validation images from `processed/images/val`.
- Slice each image into overlapping tiles.
- Run YOLO prediction on each tile.
- Convert tile predictions back to original image coordinates.
- Merge duplicate predictions across overlapping tiles.
- Export predictions in COCO-style JSON or a format that can be evaluated consistently.
- Compare metrics against normal full-image evaluation.

### Alternative
Use SAHI if installation and integration are fast enough. SAHI already implements sliced prediction and merging, and it is designed for small-object detection.

### Success criteria
- If sliced inference alone improves recall or mAP50, this is strong evidence for the report.
- If sliced inference improves recall but lowers precision, tune confidence/NMS thresholds before rejecting it.

## Experiment 3: Confidence and NMS Threshold Tuning

### Goal
Tune post-processing for the validation set, especially because dense scenes and tiny objects can be sensitive to confidence and IoU thresholds.

### Why this matters
The default evaluation settings are a reasonable start, but the dataset is not generic COCO. It has dense scenes and many tiny objects. A lower confidence threshold may recover missed objects, while a tuned NMS threshold may reduce duplicate detections.

### Grid to try

| Parameter | Values |
|---|---|
| Confidence threshold | `0.001`, `0.01`, `0.05`, `0.10`, `0.15`, `0.20`, `0.25` |
| NMS IoU threshold | `0.45`, `0.55`, `0.65`, `0.70`, `0.75` |
| Max detections | `300`, `600`, `1000` |

### Script to add
Add a small script:

```text
scripts/tune_eval_thresholds.py
```

Responsibilities:
- Run validation for selected threshold combinations.
- Save a CSV table with precision, recall, mAP50, and mAP50-95.
- Identify the best settings by mAP50-95 and by recall.

### Success criteria
- A small but defensible improvement over the current improved result.
- More important: a clean table in the final report showing that post-processing was tuned systematically.

## Experiment 4: Specialist Models or Ensembles

### Goal
Test the teacher's idea of training different YOLOs for different object groups.

### Recommended version
Do not train one model per class immediately. That would create seven models, increase inference complexity, and may overfit rare classes. Start with two specialist models:

| Specialist | Classes or samples | Reason |
|---|---|---|
| Small/dense model | images with tiny boxes or many boxes | Targets the main false-negative problem |
| Rare-class model | images containing `Aircraft`, `Drone`, `Helicopter` | Tests whether oversampling can be improved further |

### Possible implementation
- Keep labels for all classes, but oversample images matching the specialist condition.
- Train a second model using the same validation split.
- Compare the specialist model alone before attempting an ensemble.
- If useful, merge predictions from improved full model + specialist model using weighted boxes fusion.

### Success criteria
- Specialist model must improve at least one weak area without severely hurting overall mAP.
- If ensemble improves final metrics, use it as the final system.
- If not, report it as an attempted experiment and keep the sliced model as the final direction.

## Experiment 5: Focal Loss or Imbalance-Aware Loss

### Goal
Address class imbalance and hard examples more directly.

### Reality check
This is a valid idea, but it is less practical than slicing because the current training pipeline uses standard Ultralytics YOLO. Focal loss may require modifying Ultralytics internals or using a model/config that already supports focal-style classification loss.

### Recommended approach
- Do this only after slicing and threshold tuning.
- First search whether the installed Ultralytics version supports focal/varifocal loss settings for YOLO11 detection.
- If not exposed cleanly, avoid hacking the library unless there is enough time to verify correctness.

### Success criteria
- Must be compared fairly against the improved run and sliced run.
- Must not only improve rare classes while damaging overall mAP too much.

## Recommended Final Timeline

### Step 1: Fast validation of sliced inference
- Add or integrate sliced inference.
- Test `leonardo_improved-4/weights/best.pt` on validation with slicing.
- Save metrics and example predictions.
- This gives a quick answer about whether slicing helps before retraining.

### Step 2: Sliced training dataset
- Build `processed_sliced/`.
- Generate EDA summary for the sliced dataset.
- Train `leonardo_sliced_yolo11s`.
- Compare against baseline and improved model.

### Step 3: Threshold tuning
- Tune confidence, NMS IoU, and max detections for the best model.
- Save results in a table.
- Pick the final settings based on mAP50-95, with recall as secondary priority.

### Step 4: Optional specialist model
- Only run this if time and GPU budget remain.
- Prefer small/dense specialist over per-class models.
- Avoid seven separate class-specific models unless the assignment explicitly rewards architecture complexity over reliability.

### Step 5: Final report update
- Correct the task wording to object detection, not segmentation.
- Add the completed improved-run results.
- Add sliced/tiled experiment results.
- Include a clear ablation table: baseline, improved, sliced inference, sliced training, tuned final model.
- Add qualitative examples showing small/dense scenes before and after slicing.

## Proposed Final Results Table Format

Use this structure in the final report:

| Experiment | Model | Main change | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| Baseline | YOLO11n | 960 px | `0.790` | `0.642` | `0.715` | `0.382` |
| Improved | YOLO11s | 1280 px + rare oversampling | `0.836` | `0.715` | `0.792` | `0.439` |
| Sliced inference | YOLO11s | tiled validation prediction | TBD | TBD | TBD | TBD |
| Sliced training | YOLO11s | trained on crops | TBD | TBD | TBD | TBD |
| Final tuned model | best model | threshold/NMS tuned | TBD | TBD | TBD | TBD |

## Best Final Story

The strongest final project story would be:

1. We started with a clean YOLO detection baseline.
2. EDA showed that the real challenge was tiny objects and dense scenes.
3. A stronger high-resolution YOLO11s model improved all metrics.
4. Teacher feedback suggested slicing images to preserve detail.
5. We implemented sliced inference/training and showed whether it improves small-object recall.
6. We tuned post-processing and selected the final model based on validation metrics.

This story is coherent, technically justified, and directly connected to both the teacher feedback and the observed data.

## Final Recommendation

Do slicing first. It is the most valid teacher suggestion and the best match for the dataset. After that, tune thresholds. Treat specialist YOLOs and focal loss as optional experiments, not the core final direction.
