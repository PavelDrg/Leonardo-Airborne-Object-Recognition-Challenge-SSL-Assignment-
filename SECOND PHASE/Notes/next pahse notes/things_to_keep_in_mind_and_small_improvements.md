# Things To Keep In Mind And Small Improvements

This note collects extra ideas that are useful but not necessarily the main final direction. The main plan should still prioritize slicing, threshold tuning, and small/dense specialist training.

## Keep The Task Wording Correct

The dataset uses bounding boxes, so the safest wording is:

```text
object detection with bounding-box localization and class prediction
```

Avoid calling it true segmentation unless we actually create or receive pixel masks. The teacher used the word segmentation broadly, but technically our labels are detection labels.

## Always Compare Against The Improved Run

The baseline is no longer the main target. The current result to beat is:

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Improved YOLO11s | `0.836` | `0.715` | `0.792` | `0.439` |

Any new experiment should say clearly whether it beats this model or only gives a useful side result.

## Add Size-Based Evaluation

This is one of the most useful small additions for the final report.

Why:
- Our main claim is that small objects are hard.
- Global mAP does not prove whether an experiment helped small objects specifically.
- Slicing, high resolution, and small/dense specialist models should be judged on tiny/small objects.

Suggested groups:

| Group | Rule |
|---|---|
| Tiny | box area `< 0.0005` |
| Small | box area `< 0.0015` |
| Medium | `0.0015 <= area < 0.01` |
| Large | box area `>= 0.01` |
| Dense image | object count `>= 10` |
| Very dense image | object count `>= 30` |

Useful output table:

| Model | Tiny recall | Small recall | Medium recall | Large recall | Dense-image mAP50 |
|---|---:|---:|---:|---:|---:|
| Baseline | TBD | TBD | TBD | TBD | TBD |
| Improved | TBD | TBD | TBD | TBD | TBD |
| Sliced / specialist | TBD | TBD | TBD | TBD | TBD |

Even if this does not improve the model, it makes the analysis stronger.

## Tune Thresholds Per Class

The default confidence threshold may not be ideal for every class.

Possible issue:
- Rare or small classes may need lower thresholds to improve recall.
- Frequent/easier classes may tolerate higher thresholds to reduce false positives.

Classes worth checking carefully:
- `Human`
- `Obstacle`
- `GroundVehicle`
- `Aircraft`
- `Helicopter`
- `Drone`

Suggested approach:
- Start with global threshold tuning.
- If one or two classes are still weak, try class-specific thresholds only for those classes.
- Keep the method simple enough to explain in the report.

## Try Hard-Example Mining

Hard-example mining is a strong follow-up if there is time.

Idea:
1. Run the improved model on the train or validation split.
2. Identify images with many false negatives, dense scenes, or low-confidence detections.
3. Build a training list that repeats those images.
4. Train another model with hard images oversampled.

Why it may help:
- It targets actual model failures, not just class imbalance.
- It is more focused than repeating all rare-class images.
- It may improve recall on dense and cluttered scenes.

Risk:
- If hard examples come from validation, do not train on them directly. Use validation only for analysis. Hard-mining for training should use train images.

## Be Careful With One-Class YOLOs

Training one YOLO per class sounds attractive, but it has problems:

- Seven models are expensive to train and evaluate.
- Merging predictions becomes complicated.
- Some classes have limited data and may overfit.
- The final system becomes harder to explain.

Better first version:
- one small/dense specialist model
- one rare-class specialist model if time remains
- ensemble only if the specialist clearly adds useful detections

## Try Class-Aware NMS Or Weighted Box Fusion

Dense scenes can suffer from NMS suppressing nearby true objects.

Things to test:
- higher NMS IoU threshold
- class-aware NMS
- weighted boxes fusion for sliced predictions
- higher `max_det`, for example `1000`

This matters especially for:
- crowded `Human` scenes
- dense `GroundVehicle` scenes
- overlapping `Obstacle` detections

## Try Test-Time Augmentation

Test-time augmentation can sometimes improve detection without retraining.

Possible options:
- horizontal flip
- multi-scale inference
- tiled inference with different tile sizes

Risk:
- It increases inference time.
- It may add duplicates unless merging is handled carefully.

Use it as an inference experiment, not as the main project direction.

## Consider YOLO11m Only If Time Allows

The improved run used YOLO11s and already took about `16` hours. YOLO11m may improve performance, but it will likely take longer.

Use YOLO11m only if:
- both machines are free enough
- slicing/specialist experiments are already running or finished
- we can afford a long training run with uncertain payoff

Possible command idea:

```bash
python scripts/train_yolo.py --data "processed/data.yaml" --variant improved --project "runs/detect" --name "leonardo_yolo11m_test"
```

This would require editing `train_yolo.py` or adding a new variant to use `yolo11m.pt`.

## Consider Higher Resolution Carefully

Going above `1280` may help tiny objects, but it is expensive.

Possible values:
- `1536`
- `1600`

Risks:
- much slower training
- smaller batch size
- GPU memory issues
- diminishing returns compared with slicing

Slicing is probably a better way to preserve detail than simply increasing full-image resolution again.

## Check Label Quality Around Borders

The preprocessing already clipped boxes and dropped invalid boxes, but border boxes remain difficult.

Useful checks:
- examples where boxes touch image boundaries
- very tiny boxes after clipping
- labels with width or height close to zero
- dense images with hundreds of boxes

If many errors come from bad or ambiguous labels, mention this as a limitation rather than overfitting to them.

## Add More Qualitative Failure Examples

The final report should not only show successful predictions.

Include:
- one easy success case
- one dense scene where improved model still misses objects
- one case where slicing/specialist model helps
- one remaining failure case

This makes the project look more honest and technically mature.

## Keep Experiments Reproducible

For every run, save:

- command used
- run folder
- checkpoint path
- metrics table
- training time
- what changed from previous run
- whether it should be kept for the final report

Suggested log row:

| Date | Owner | Experiment | Command | Runtime | Precision | Recall | mAP50 | mAP50-95 | Decision |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| TBD | A/B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | keep/drop |

## Do Not Overcomplicate The Final System

The final system should be explainable in a few sentences.

Good final story:
- baseline YOLO
- improved high-resolution YOLO
- teacher-inspired slicing or specialist sampling
- final validation comparison

Risky final story:
- many independent tricks
- unclear validation protocol
- multiple models with no clean ablation
- visual examples without metrics

## Recommended Priority List

If time becomes tight, use this order:

1. Size-based evaluation.
2. Sliced inference.
3. Threshold/NMS tuning.
4. Small/dense specialist model.
5. Hard-example mining.
6. Sliced training.
7. Ensemble/weighted boxes fusion.
8. YOLO11m or higher-resolution full-image training.
9. Focal loss research.
10. One-class YOLOs.

## Final Reminder

The best improvement is not necessarily the most complex one. The best final result is the one that:

- improves at least one important metric,
- directly addresses the tiny-object failure mode,
- can be compared fairly against the improved run,
- and can be explained clearly in the report.
