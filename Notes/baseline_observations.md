# Baseline Observations

## Training quality
- Training is stable and converges smoothly across all 18 epochs.
- Precision, recall, mAP50, and mAP50-95 all improve steadily.
- Final baseline metrics are roughly:
  - precision: 0.791
  - recall: 0.639
  - mAP50: 0.713
  - mAP50-95: 0.381

## Explicit validation summary
- The standalone eval script confirms the same overall quality:
  - precision: 0.790
  - recall: 0.642
  - mAP50: 0.715
  - mAP50-95: 0.382
- This matches the training-time validation almost exactly, so the checkpoint is consistent.

## Plot-level observations
- PR curves show the easiest classes are `Aircraft` and `Drone`.
- Harder classes are `Human`, `Obstacle`, and `GroundVehicle`.
- The confusion matrix shows many missed objects are predicted as background.
- This is expected for tiny objects and dense scenes.

## Class-level validation results
- Strong classes:
  - `Aircraft` mAP50 ~ 0.831
  - `Drone` mAP50 ~ 0.813
  - `Helicopter` mAP50 ~ 0.786
  - `Ship` mAP50 ~ 0.739
- Weaker classes:
  - `GroundVehicle` mAP50 ~ 0.673
  - `Obstacle` mAP50 ~ 0.599
  - `Human` mAP50 ~ 0.563
- The recall gap suggests the model is conservative and misses small or cluttered objects.

## Visual inspection of predictions
- Large, clear objects are detected well:
  - aircraft on runway
  - ships on water
  - obvious ground vehicles
- The model struggles more when:
  - objects are very small
  - multiple objects overlap
  - contrast is low
  - the scene is crowded

## Main failure modes
- false negatives for tiny objects
- duplicate detections in dense scenes
- class confusion in crowded scenes
- background suppression too aggressive for some small targets

## Conclusion for the baseline
- This is a solid, non-trivial baseline for the assignment.
- It already solves the task in a meaningful way and gives good evidence for a report.
- The next best step is to train a stronger model with higher resolution and more imbalance-aware sampling.

## Improved-run comparison
- The improved run is clearly better than the baseline on every main metric.
- Final improved validation metrics:
  - precision: 0.834
  - recall: 0.716
  - mAP50: 0.792
  - mAP50-95: 0.439
- Compared with the baseline, the biggest gains are in recall and localization quality.
- Class-level mAP50 also improved across the board, especially for the hard classes.
- This suggests that higher resolution plus rare-class oversampling helped the model learn the small and rare objects better.
- The strongest classes remain Aircraft, Drone, and Helicopter, but GroundVehicle, Human, and Obstacle improved too.
- The confusion matrix still shows background misses, but less than in the baseline.

## Useful improvement ideas
- increase input resolution for the stronger model
- try tiled inference / SAHI for small objects
- tune confidence and NMS thresholds
- oversample images containing rare classes
- add more aggressive small-object augmentation
