# Two-Person Parallel Plan

## Situation

We have around `60` hours left and two separate machines/GPUs. That means we should not run one shared sequential experiment chain. Each person should own an independent track that can produce useful results alone, while still using the same validation split and final metric table.

Current best model:

| Run | Model | Main change | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| Baseline | `yolo11n.pt` | 960 px | `0.790` | `0.642` | `0.715` | `0.382` |
| Improved | `yolo11s.pt` | 1280 px + rare oversampling | `0.836` | `0.715` | `0.792` | `0.439` |

The final experiments must beat or at least explain results relative to the improved run, not just the baseline.

## Shared Rules

Both people must follow the same validation protocol.

Shared validation data:

```text
processed/data.yaml
processed/images/val
processed/labels/val
```

Shared reference checkpoint:

```text
runs/detect/leonardo_improved-4/weights/best.pt
```

Shared target metrics:

| Metric | Target to beat |
|---|---:|
| Precision | `0.836` |
| Recall | `0.715` |
| mAP50 | `0.792` |
| mAP50-95 | `0.439` |

Shared reporting format:

| Experiment | Owner | Model | Main change | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| Improved reference | both | YOLO11s | 1280 + rare oversampling | `0.836` | `0.715` | `0.792` | `0.439` | current best |
| Person A result | A | TBD | slicing/tiled inference/training | TBD | TBD | TBD | TBD | TBD |
| Person B result | B | TBD | scale-specialist/training/tuning | TBD | TBD | TBD | TBD | TBD |
| Final selected | both | TBD | best validated approach | TBD | TBD | TBD | TBD | final model |

Shared naming convention:

```text
runs/detect/leonardo_A_<experiment_name>
runs/detect/leonardo_B_<experiment_name>
```

Shared notes convention:

```text
SECOND PHASE/Notes/next pahse notes/person_A_log.md
SECOND PHASE/Notes/next pahse notes/person_B_log.md
```

Each person should keep a short experiment log with command, start time, end time, checkpoint path, final metrics, and whether the result is worth keeping.

## Division of Work

| Person | Main track | Why this split is good |
|---|---|---|
| Person A | Slicing and tiled inference/training | Directly addresses the teacher's strongest suggestion and the dataset's tiny-object problem |
| Person B | Scale-aware/specialist training plus threshold tuning | Tests the teacher's separate-YOLO idea and searches for cheap post-processing gains |

The two tracks are independent enough to run at the same time on separate GPUs. They only need to coordinate on evaluation format and final comparison.

## Person A Track: Slicing / Tiling

### Main Question

Does preserving local resolution through image slicing improve small-object recall and localization?

### Why Person A Should Prioritize This

This is the most defensible next experiment because:

- The teacher explicitly suggested cutting images into slices.
- The dataset has many tiny boxes; median box area is about `0.0015` of the image.
- Dense scenes cause missed detections and background errors.
- The improved full-image model helped, but still loses detail through resizing.

### A1: Sliced Inference on Existing Improved Model

This should be Person A's first experiment because it avoids retraining.

Goal:

- Use `leonardo_improved-4/weights/best.pt`.
- Slice validation images into overlapping crops.
- Run YOLO on each crop.
- Convert predictions back to full-image coordinates.
- Merge overlapping predictions.
- Compare against normal validation.

Suggested tile grid:

| Tile size | Overlap | Why |
|---:|---:|---|
| `640` | `20%` | Fast first test |
| `640` | `30%` | More border protection |
| `768` | `25%` | Good balance |
| `960` | `25%` | Larger context, still more local than full image |

Suggested confidence values:

| Parameter | Values |
|---|---|
| Confidence | `0.01`, `0.05`, `0.10`, `0.15`, `0.20` |
| NMS IoU | `0.45`, `0.55`, `0.65`, `0.70` |
| Max detections | `600`, `1000` |

Expected runtime:

| Scope | Estimated time |
|---|---:|
| One sliced inference setting | `1-3h` |
| Small grid | `6-12h` |
| Large grid | `12-24h` |

Deliverables:

- `scripts/evaluate_sliced_yolo.py` or documented SAHI command.
- CSV of sliced inference metrics.
- Best tile/overlap/conf/NMS setting.
- Example prediction images for dense scenes.
- Short note: `person_A_sliced_inference_results.md`.

Success criteria:

- Recall above `0.715`, or mAP50 above `0.792`, or mAP50-95 above `0.439`.
- If precision drops but recall improves strongly, keep the result as useful and try threshold tuning.

### A2: Sliced Dataset Generation

Only start this after at least one sliced inference setting looks promising, or if inference integration is blocked.

Goal:

- Build a new YOLO dataset where each original image is split into tiles.
- Preserve the existing image-level split to avoid leakage.

Output folder:

```text
processed_sliced/
```

Script to create:

```text
scripts/preprocess_sliced_dataset.py
```

Initial settings:

| Parameter | Value |
|---|---:|
| Tile size | `768` |
| Overlap | `25%` |
| Minimum retained box area | `25%` |
| Empty tile ratio | cap empty tiles to avoid too many negatives |

Important implementation rule:

- Tiles from train images go only to train.
- Tiles from validation images go only to validation.
- Do not randomly split tiles after slicing.

Deliverables:

- `processed_sliced/data.yaml`
- `processed_sliced/meta/summary.json`
- tile count and class distribution table
- several sample tile images with labels

Expected runtime:

| Step | Estimated time |
|---|---:|
| Script implementation | `4-8h` |
| Dataset generation | `1-3h` |
| Sanity checking labels | `1-2h` |

### A3: Sliced Training

Start this only if there is enough GPU time after A1/A2.

First model:

```bash
python scripts/train_yolo.py --data "processed_sliced/data.yaml" --variant improved --project "runs/detect" --name "leonardo_A_sliced_yolo11s"
```

Possible adjustment:

- If training is too slow, use YOLO11n first as a faster proof of concept.
- If memory is tight, reduce image size for sliced tiles because the crops already zoom into objects.

Estimated runtime:

| Model | Estimated time |
|---|---:|
| YOLO11n sliced | `8-18h` |
| YOLO11s sliced | `20-40h` |

Person A should not spend the whole 60 hours only training if sliced inference already gives useful results. The final report benefits more from a clean ablation than from one uncontrolled long run.

## Person B Track: Specialist Models and Threshold Tuning

### Main Question

Can scale-aware training or post-processing improve the current improved model without needing full slicing?

### Why Person B Should Prioritize This

This covers the teacher's second suggestion about different YOLOs, but keeps it realistic. Instead of training seven per-class models, Person B should start with scale/density specialists.

### B1: Threshold and NMS Tuning

This is the fastest useful experiment and should be Person B's first task.

Goal:

- Use `leonardo_improved-4/weights/best.pt`.
- Evaluate different confidence, NMS IoU, and max detection settings.
- Find whether the current model is under-detecting because post-processing is too conservative.

Grid:

| Parameter | Values |
|---|---|
| Confidence | `0.001`, `0.01`, `0.05`, `0.10`, `0.15`, `0.20`, `0.25` |
| NMS IoU | `0.45`, `0.55`, `0.65`, `0.70`, `0.75` |
| Max detections | `300`, `600`, `1000` |

Script to create:

```text
scripts/tune_eval_thresholds.py
```

Deliverables:

- `runs/threshold_tuning/improved_threshold_grid.csv`
- best setting by mAP50-95
- best setting by recall
- short note: `person_B_threshold_results.md`

Expected runtime:

| Scope | Estimated time |
|---|---:|
| Small grid | `2-5h` |
| Larger grid | `5-10h` |

Success criteria:

- Any improvement over `0.439` mAP50-95 is valuable.
- If recall improves above `0.715` with a small precision loss, this may still be useful for small-object detection.

### B2: Object-Size Analysis

Goal:

- Categorize training images by object size and density.
- Build lists for small/dense and large/clear training experiments.

Definitions to start with:

| Category | Rule |
|---|---|
| Small-object image | at least one box with area `< 0.0015` |
| Very small-object image | at least one box with area `< 0.0005` |
| Dense image | object count `>= 10` |
| Very dense image | object count `>= 30` |
| Large-object image | median box area `> 0.01` |

Script to create:

```text
scripts/build_specialist_train_lists.py
```

Outputs:

```text
processed/meta/train_small_dense_x2.txt
processed/meta/train_small_dense_x3.txt
processed/meta/train_large_clear.txt
processed/data_small_dense.yaml
processed/data_large_clear.yaml
```

Expected runtime:

| Step | Estimated time |
|---|---:|
| Analysis/list script | `2-4h` |
| Summary plots/tables | `1-2h` |

### B3: Small/Dense Specialist Training

This should be the first specialist training, not one-model-per-class.

Goal:

- Train a model that sees small/dense images more often.
- Keep all classes in labels, but sample small/dense images more frequently.

Recommended command:

```bash
python scripts/train_yolo.py --data "processed/data_small_dense.yaml" --variant improved --project "runs/detect" --name "leonardo_B_small_dense_yolo11s"
```

If time is tight:

```bash
python scripts/train_yolo.py --data "processed/data_small_dense.yaml" --variant baseline --project "runs/detect" --name "leonardo_B_small_dense_yolo11n"
```

Estimated runtime:

| Model | Estimated time |
|---|---:|
| YOLO11n small/dense | `8-18h` |
| YOLO11s small/dense | `14-28h` |

Success criteria:

- Overall mAP close to improved run, with better recall.
- Better qualitative detections on small/dense validation examples.
- If overall mAP is lower but recall is higher, the model may still be useful in an ensemble.

### B4: Large/Clear Specialist Training

This is optional and lower priority.

Reason:

- The baseline already handles large/clear objects reasonably well.
- The main remaining weakness is small objects.

Only run this if B1 and B3 finish early.

Estimated runtime:

| Model | Estimated time |
|---|---:|
| YOLO11n large/clear | `8-15h` |
| YOLO11s large/clear | `12-25h` |

### B5: Ensemble Test

If the small/dense specialist improves recall but hurts precision, test combining it with the improved full model.

Candidates:

| Ensemble | Purpose |
|---|---|
| Improved full model + small/dense specialist | recover missed tiny/dense objects |
| Improved full model + sliced inference result from Person A | combine best full-image and local-resolution predictions |

Merging methods:

- standard NMS
- class-aware NMS
- weighted boxes fusion if available

Expected runtime:

| Step | Estimated time |
|---|---:|
| Implement merge/evaluate | `4-8h` |
| Run validation merge | `2-6h` |

## Shared 60-Hour Timeline

Because both people have separate GPUs, long runs can overlap.

| Time window | Person A | Person B | Sync point |
|---|---|---|---|
| `0-4h` | Set up sliced inference path or SAHI test | Set up threshold tuning script | Agree on metric output CSV columns |
| `4-12h` | Run first sliced inference settings | Run threshold/NMS grid on improved model | Share first results and decide whether thresholds matter |
| `12-20h` | Expand best sliced inference grid | Build object-size/density train lists | Compare best quick experiments |
| `20-30h` | Build `processed_sliced/` if slicing helps | Start small/dense specialist training | Decide which long training has priority for final story |
| `30-45h` | Train sliced model or finalize sliced inference results | Continue/finish small-dense training | Start collecting figures and qualitative examples |
| `45-54h` | Evaluate sliced model and save examples | Evaluate specialist model and optional ensemble | Pick final model/approach |
| `54-60h` | Write slicing section/results | Write specialist/threshold section/results | Merge final report table and conclusion |

## Decision Points

### After 12 Hours

Pick one of these:

| Result | Decision |
|---|---|
| Sliced inference improves recall/mAP | Person A continues slicing and maybe sliced training |
| Threshold tuning improves mAP50-95 | Person B keeps tuning and prepares final post-processing result |
| Neither quick result helps | Person A still tries sliced dataset; Person B prioritizes small/dense specialist |

### After 30 Hours

Pick final long-run emphasis:

| Situation | Best action |
|---|---|
| Sliced inference clearly helps | Train sliced YOLO or finalize sliced inference as final method |
| Small/dense specialist clearly helps | Test ensemble with improved full model |
| Both help | Try merged predictions if time allows |
| Neither helps | Keep improved YOLO11s as final model and report failed experiments honestly |

### After 50 Hours

Stop starting new long experiments. Only evaluate, copy figures, and update the report.

## Minimum Deliverables From Each Person

Person A must deliver:

- sliced inference/training command or script
- metrics table against the improved run
- at least two qualitative figures showing dense/small scenes
- short conclusion: did slicing help or not?

Person B must deliver:

- threshold tuning table
- object-size/density analysis table
- specialist training metrics if training completed
- short conclusion: did specialist sampling help or not?

Shared final deliverables:

- one final comparison table
- one final selected model or method
- one paragraph explaining why it was selected
- one limitations paragraph
- report-ready images copied into a local report/notes images folder

## Recommended Final Comparison Table

Use this table in the final report or notes:

| Experiment | Owner | Precision | Recall | mAP50 | mAP50-95 | Runtime | Keep? |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline YOLO11n | previous | `0.790` | `0.642` | `0.715` | `0.382` | previous | yes, reference |
| Improved YOLO11s | previous | `0.836` | `0.715` | `0.792` | `0.439` | `~16h` | yes, current best |
| Sliced inference | A | TBD | TBD | TBD | TBD | TBD | TBD |
| Sliced training | A | TBD | TBD | TBD | TBD | TBD | TBD |
| Threshold tuned improved | B | TBD | TBD | TBD | TBD | TBD | TBD |
| Small/dense specialist | B | TBD | TBD | TBD | TBD | TBD | TBD |
| Ensemble/final | both | TBD | TBD | TBD | TBD | TBD | final candidate |

## What Not To Do

- Do not train seven separate one-class YOLOs first. It is too expensive and hard to merge cleanly.
- Do not modify Ultralytics internals for focal loss until slicing and specialist training are tested.
- Do not compare experiments on different validation splits.
- Do not rely only on visual examples; every result needs the same four metrics.
- Do not start a new long training run in the last `10` hours.

## Best Use of Two GPUs

The best use of the two separate machines is:

| GPU | First priority | Second priority | Optional |
|---|---|---|---|
| Person A GPU | sliced inference grid | sliced dataset + sliced model | sliced/full ensemble |
| Person B GPU | threshold grid | small/dense specialist | large/clear specialist or focal-loss research |

This gives two independent chances to beat the improved model while keeping the final story coherent: one track follows the teacher's slicing idea, and the other tests the teacher's separate-model idea in a controlled way.

## Final Recommendation

Person A should own slicing. Person B should own threshold tuning and small/dense specialist training. After `30` hours, compare early results and decide whether the final system should be sliced inference, sliced training, a specialist model, or an ensemble. After `50` hours, stop experimenting and focus on evaluation, figures, and the final report.
