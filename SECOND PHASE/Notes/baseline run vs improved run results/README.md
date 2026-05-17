# Second Phase Notes

These notes cover the completed improved YOLO run and are meant to replace the old report wording where the improved run was still described as ongoing.

## Files
- `improved_run_observations.md` - detailed notes on what changed, how the improved run trained, what the metrics show, and what helped.
- `improved_vs_baseline.md` - direct comparison against the baseline numbers used in the first report version.
- `improved_run_figures.md` - list of the new graphs, images, and validation examples to use in the updated report or presentation.

## Main conclusion
- The improved run is clearly better than the baseline on all main detection metrics.
- The biggest gains are in recall and mAP50-95, which matches the goal of improving small-object detection and localization.
- The cost is much higher training time because the model is larger and the input resolution is higher.

## Best improved run
- Run folder: `runs/detect/leonardo_improved-4/`
- Weights: `runs/detect/leonardo_improved-4/weights/best.pt`
- Model: `yolo11s.pt`
- Image size: `1280`
- Epochs: `20`
- Training data: `processed/data_improved_oversampled.yaml`
- Main metrics: precision `0.836`, recall `0.715`, mAP50 `0.792`, mAP50-95 `0.439`
