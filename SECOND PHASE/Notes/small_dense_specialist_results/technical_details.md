# Technical Details - Small/Dense Specialist Experiment

## Purpose

This document explains the technical side of the small/dense specialist experiment: what changed in the code, how the training list was generated, how the run can be reproduced, and what stayed unchanged from the previous pipeline.

The experiment does not change the YOLO architecture and does not modify Ultralytics internals. The main technical change is the training sampler.

## Pipeline Summary

The final pipeline is:

```text
raw Kaggle data
  -> scripts/preprocess_dataset.py
  -> processed/data.yaml
  -> scripts/build_specialist_train_lists.py
  -> processed/data_small_dense.yaml
  -> scripts/train_yolo.py --variant improved
  -> runs/detect/leonardo_B_small_dense_x2_yolo11s
```

The trained model is still `YOLO11s`, with `1280` image size and the same broad training setup as the previous improved run. The difference is that the train set contains repeated entries for images with small objects or dense object layouts.

## Added Code Files

### `scripts/build_specialist_train_lists.py`

This is the main new script for the small/dense experiment.

Responsibilities:

- read `processed/meta/train_manifest.csv`;
- read YOLO label files from `processed/labels/train`;
- compute image-level statistics:
  - object count;
  - minimum bounding-box area;
  - median bounding-box area;
  - whether the image contains small objects;
  - whether the image is dense;
  - whether the image contains rare classes;
- build repeated-image train lists;
- write new Ultralytics-compatible `.yaml` files.

Rules used:

| Category | Rule |
|---|---|
| Tiny image | at least one box with normalized area `< 0.0005` |
| Small image | at least one box with normalized area `< 0.0015` |
| Dense image | at least `10` objects |
| Very dense image | at least `30` objects |
| Rare image | contains `Aircraft`, `Drone`, or `Helicopter` |
| Small/dense image | small image or dense image |

Generated outputs:

```text
processed/data_small_dense.yaml
processed/data_small_dense_x2.yaml
processed/data_small_dense_x3.yaml
processed/data_large_clear.yaml
processed/meta/train_small_dense_x2_rare_x2.txt
processed/meta/train_small_dense_x3_rare_x2.txt
processed/meta/train_large_clear.txt
processed/meta/specialist_summary.json
processed/meta/specialist_image_stats.csv
```

The final run used:

```text
processed/data_small_dense.yaml
```

This points to:

```text
processed/meta/train_small_dense_x2_rare_x2.txt
```

Training list size after repetition:

```text
21964 image entries
```

Original unique training images:

```text
14145 images
```

## Sampling Logic

Each image starts with repeat factor `1`.

If an image is small/dense:

```text
repeat >= 2
```

If an image contains rare classes:

```text
repeat >= 2
```

This means an image is repeated when it is important either for small/dense detection or for rare-class coverage. The method preserves the rare-class oversampling idea from the improved run while adding explicit focus on the dataset's hardest examples.

Important detail: labels are not filtered or modified. All classes remain present in each image. This is still a standard multi-class object detection model, not a set of one-class detectors.

## Changes In `scripts/train_yolo.py`

The existing training script was extended with command-line overrides:

```text
--imgsz
--epochs
--batch
--workers
--patience
```

Reason:

- the local GPU has 8 GB VRAM;
- the improved config uses `YOLO11s`, `imgsz=1280`, and `batch=6`;
- if CUDA runs out of memory, the run can be restarted with `--batch 4` or `--batch 2` without editing the script.

Fallback example:

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo.py --data "processed\data_small_dense.yaml" --variant improved --project "runs\detect" --name "leonardo_B_small_dense_x2_yolo11s_b4" --device 0 --batch 4
```

The final run completed with the default improved batch size:

```text
batch=6
```

## Optional Evaluation Utility: `scripts/tune_eval_thresholds.py`

This script was prepared for threshold/NMS tuning.

Responsibilities:

- run validation across multiple confidence thresholds;
- test multiple NMS IoU thresholds;
- test multiple `max_det` values;
- save the results in a CSV table.

It was not part of the small/dense training itself. It is useful after a checkpoint exists, for example the improved reference checkpoint or the new small/dense checkpoint.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\tune_eval_thresholds.py --weights "best.pt" --data "processed\data.yaml" --device 0 --name "improved_threshold_quick" --conf-values "0.001,0.01,0.05,0.10" --iou-values "0.55,0.70,0.75" --max-det-values "600,1000"
```

The improved reference checkpoint was later evaluated with this grid:

```powershell
.\.venv\Scripts\python.exe scripts\tune_eval_thresholds.py --weights "improved_reference_best.pt" --data "processed\data.yaml" --device 0 --name "improved_reference_threshold_quick" --conf-values "0.001,0.01,0.05,0.10" --iou-values "0.55,0.70,0.75" --max-det-values "600,1000"
```

Grid size:

```text
4 confidence values x 3 IoU values x 2 max_det values = 24 validation runs
```

Best mAP50-95 setting:

| conf | IoU | max_det | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| `0.001` | `0.70` | `1000` | `0.8372` | `0.7164` | `0.7935` | `0.4401` |

Best recall setting:

| conf | IoU | max_det | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| `0.001` | `0.55` | `600` | `0.8422` | `0.7405` | `0.8043` | `0.4371` |

Conclusion from threshold tuning: post-processing alone improves the reference model slightly, but the small/dense specialist remains stronger overall.

## Preprocessing Command

The raw dataset was extracted from `archive.zip`, then converted to YOLO format with:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --output-root "processed" --force
```

Result:

```text
train images: 14145
val images: 3536
train background images: 220
val background images: 55
classes: Aircraft, Drone, GroundVehicle, Helicopter, Human, Obstacle, Ship
```

Preprocessing keeps the split at image level, which prevents train/validation leakage.

## Small/Dense List Generation Command

```powershell
.\.venv\Scripts\python.exe scripts\build_specialist_train_lists.py --processed-root "processed"
```

Generated summary:

```text
tiny_images: 2994
small_images: 5657
dense_images: 1254
very_dense_images: 282
rare_images: 3985
small_dense_images: 5871
large_clear_images: 4793
```

## Training Command

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo.py --data "processed\data_small_dense.yaml" --variant improved --project "runs\detect" --name "leonardo_B_small_dense_x2_yolo11s" --device 0
```

Relevant settings from the improved variant:

| Setting | Value |
|---|---:|
| weights | `yolo11s.pt` |
| imgsz | `1280` |
| epochs | `20` |
| batch | `6` |
| mosaic | `1.0` |
| mixup | `0.10` |
| copy_paste | `0.15` |
| max_det | `600` |
| lr0 | `0.003` |
| close_mosaic | `15` |

Training time:

```text
15.773 hours
```

## Validation

Validation used the original validation split:

```text
processed/data.yaml
```

This is important because it makes the comparison fair against the previous baseline and improved reference.

Final metrics:

| Metric | Value |
|---|---:|
| Precision | `0.845` |
| Recall | `0.746` |
| mAP50 | `0.814` |
| mAP50-95 | `0.447` |

## What Did Not Change

The experiment did not change:

- YOLO architecture;
- Ultralytics internals;
- loss function;
- label format;
- class list;
- validation split;
- evaluation metrics;
- bounding-box coordinates;
- the original preprocessing pipeline.

Focal loss was not used. It remains a possible future direction, but it would require riskier changes to the training internals.

Seven separate one-class YOLO models were not trained. The experiment uses a simpler and more explainable version of the specialist-model idea: one multi-class detector trained with small/dense-aware sampling.

## Conceptual Difference From The Improved Reference

Improved reference:

```text
YOLO11s + 1280 px + rare-class oversampling
```

Small/dense specialist:

```text
YOLO11s + 1280 px + small/dense sampling + rare-class preservation
```

The tested question is:

```text
Does the model improve if it sees small-object and dense-scene images more often during training?
```

The validation results suggest yes: all four main metrics improved over the improved reference, with the largest absolute gain in recall.

## Why The Result Matters

The dataset contains many small objects, and the baseline showed that the main failure mode was missed detections/background errors. Small/dense sampling targets that failure mode directly.

Gain over the improved reference:

| Metric | Gain |
|---|---:|
| Precision | `+0.0095` |
| Recall | `+0.0307` |
| mAP50 | `+0.0216` |
| mAP50-95 | `+0.0078` |

The recall gain is the most important one because the goal was to recover more missed objects in difficult scenes.

## Reproduction Commands

To reproduce the experiment from the raw extracted dataset:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --output-root "processed" --force
.\.venv\Scripts\python.exe scripts\build_specialist_train_lists.py --processed-root "processed"
.\.venv\Scripts\python.exe scripts\train_yolo.py --data "processed\data_small_dense.yaml" --variant improved --project "runs\detect" --name "leonardo_B_small_dense_x2_yolo11s" --device 0
```

For explicit evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_yolo.py --weights "runs\detect\leonardo_B_small_dense_x2_yolo11s\weights\best.pt" --data "processed\data.yaml" --project "runs\val" --name "leonardo_B_small_dense_x2_yolo11s"
```

## Slide-Ready Technical Summary

```text
We kept the YOLO11s architecture and 1280 px setup, but changed the training sampler. Images containing small bounding boxes or dense object layouts were repeated in the training list, while rare-class oversampling was preserved. This produced a small/dense specialist model and improved recall from 0.7153 to 0.746 on the same validation split.
```
