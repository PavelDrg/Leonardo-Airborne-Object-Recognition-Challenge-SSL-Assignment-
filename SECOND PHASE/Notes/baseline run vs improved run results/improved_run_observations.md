# Improved Run Observations

## Context
In the first report version, the improved run was still in progress. It has now finished successfully. The final completed run is `leonardo_improved-4`, stored under `runs/detect/leonardo_improved-4/`.

The baseline already showed that the dataset was learnable, but it also showed the main weakness: many missed small objects, especially in crowded or low-contrast scenes. Because of that, the improved run was designed to improve recall and localization rather than just maximize precision.

## What changed from the baseline

| Setting | Baseline | Improved run | Reason |
|---|---:|---:|---|
| Model | `yolo11n.pt` | `yolo11s.pt` | More capacity for small and dense objects |
| Input size | `960` | `1280` | Small objects occupy more pixels after resizing |
| Epochs | `18` | `20` | Slightly longer training budget |
| Batch size | `16` | `6` | Required by the higher resolution and larger model |
| Max detections | `300` | `600` | Dense scenes can contain many objects |
| Initial LR | `0.01` | `0.003` | More conservative optimization for stronger run |
| Mosaic | `0.8` | `1.0` | More aggressive spatial augmentation |
| MixUp | `0.05` | `0.10` | More augmentation diversity |
| Copy-paste | `0.0` | `0.15` | More synthetic object variation |
| Rare-class sampling | none | rare images oversampled x2 | Helps `Aircraft`, `Drone`, and `Helicopter` appear more often during training |

The command used for this run was:

```bash
python scripts/train_yolo.py --data "processed/data.yaml" --variant improved --oversample-rare --oversample-factor 2 --project "runs/detect"
```

The training script generated `processed/data_improved_oversampled.yaml`, which points YOLO to `processed/meta/train_oversampled_x2.txt`. The oversampled list contains repeated paths for images marked as rare in `train_manifest.csv`.

## Training quality
- Training completed all `20` epochs.
- Precision, recall, mAP50, and mAP50-95 improved steadily over the run.
- The final epoch was also the best point in the saved CSV, so there is no sign that training collapsed near the end.
- The stronger run took much longer than the baseline because it used a larger model and `1280` image size.

Final improved metrics from `runs/detect/leonardo_improved-4/results.csv`:

| Metric | Value |
|---|---:|
| Precision | `0.83553` |
| Recall | `0.71526` |
| mAP50 | `0.79243` |
| mAP50-95 | `0.43921` |

Rounded values for the report:

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Improved validation | `0.836` | `0.715` | `0.792` | `0.439` |

## Final losses

| Loss | Baseline final | Improved final | Observation |
|---|---:|---:|---|
| Train box loss | `1.62375` | `1.48377` | Improved run fits boxes better on training data |
| Train cls loss | `1.05662` | `0.86568` | Improved run learns class separation better |
| Train dfl loss | `1.19881` | `1.21026` | Similar, slightly higher for improved run |
| Val box loss | `1.66162` | `1.61697` | Better validation box regression |
| Val cls loss | `1.06647` | `0.96719` | Better validation classification |
| Val dfl loss | `1.21389` | `1.27554` | Higher, but aggregate mAP still improved |

The important point is that the validation metrics improved even though one loss component is higher. For the report, the detection metrics are more meaningful than comparing individual loss terms directly.

## What helped
- Higher input resolution helped because the median bounding-box area is only around `0.0015` of the image. At `1280`, small targets preserve more visible detail than at `960`.
- The larger `yolo11s` model had more capacity to learn small-object features and crowded scenes.
- Rare-class oversampling increased how often images containing `Aircraft`, `Drone`, and `Helicopter` appeared during training.
- Stronger augmentation likely improved robustness, especially in scenes with different object layouts.
- Increasing `max_det` from `300` to `600` better matches the dataset because a few images contain very dense object layouts.

## Remaining issues
- The confusion matrix still shows that many errors are missed detections/background errors rather than random class swaps.
- Small, overlapping, or low-contrast objects remain the hardest cases.
- The improved run is better, but it is also much more expensive to train.
- Exact improved per-class mAP values are not stored in `results.csv`; if we need a numeric per-class table in the final report, we should rerun `evaluate_yolo.py` and save the console table.

## Short report wording
The improved run used a larger YOLO11s model, higher `1280` input resolution, stronger augmentation, a higher maximum detection limit, and rare-class oversampling. This directly targeted the baseline failure modes: small objects, dense scenes, and underrepresented classes. Compared with the baseline, the improved model increased precision from `0.790` to `0.836`, recall from `0.642` to `0.715`, mAP50 from `0.715` to `0.792`, and mAP50-95 from `0.382` to `0.439`. The strongest improvement is in recall and mAP50-95, suggesting that the model finds more objects and localizes them better.
