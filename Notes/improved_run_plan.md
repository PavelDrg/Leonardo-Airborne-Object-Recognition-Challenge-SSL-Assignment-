# Improved Run Plan

## Differences from baseline
- Uses `yolo11s.pt` instead of `yolo11n.pt`.
- Trains at `1280` resolution.
- Oversamples rare-class images in the train split.
- Uses stronger augmentation and shorter training to fit the time budget.

## Why this should help
- Rare classes are underrepresented in the raw data.
- Tiny objects need more pixels to be visible.
- The baseline already learns well, so the improved run should focus on recall and difficult classes.

## Command
`python scripts/train_yolo.py --data "processed/data.yaml" --variant improved --oversample-rare --oversample-factor 2 --project "runs/detect"`

## Expected effect
- better recall on `Human`, `Obstacle`, and `GroundVehicle`
- less background suppression for tiny objects
- more robust performance in dense scenes
