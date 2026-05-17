# Improved Run Figures and Artifacts

This file lists the new improved-run artifacts that can be used in the updated report or presentation. The images were copied into `SECOND PHASE/Notes/images/` so this note can be opened directly without depending on long paths outside this folder.

## Main plots

Training curves:

![Improved training curves](images/improved_results.png)

Precision-recall curve:

![Improved PR curve](images/improved_BoxPR_curve.png)

Precision-confidence curve:

![Improved precision curve](images/improved_BoxP_curve.png)

Recall-confidence curve:

![Improved recall curve](images/improved_BoxR_curve.png)

F1-confidence curve:

![Improved F1 curve](images/improved_BoxF1_curve.png)

Normalized confusion matrix:

![Improved normalized confusion matrix](images/improved_confusion_matrix_normalized.png)

Raw confusion matrix:

![Improved raw confusion matrix](images/improved_confusion_matrix.png)

Training label summary:

![Improved label summary](images/improved_labels.jpg)

## Validation predictions

Validation batch 0 predictions:

![Validation batch 0 predictions](images/improved_val_batch0_pred.jpg)

Validation batch 0 labels:

![Validation batch 0 labels](images/improved_val_batch0_labels.jpg)

Validation batch 1 predictions:

![Validation batch 1 predictions](images/improved_val_batch1_pred.jpg)

Validation batch 1 labels:

![Validation batch 1 labels](images/improved_val_batch1_labels.jpg)

Validation batch 2 predictions:

![Validation batch 2 predictions](images/improved_val_batch2_pred.jpg)

Validation batch 2 labels:

![Validation batch 2 labels](images/improved_val_batch2_labels.jpg)

## Training batch examples

Early training batch 0:

![Train batch 0](images/improved_train_batch0.jpg)

Early training batch 1:

![Train batch 1](images/improved_train_batch1.jpg)

Early training batch 2:

![Train batch 2](images/improved_train_batch2.jpg)

Late training batch 15110:

![Train batch 15110](images/improved_train_batch15110.jpg)

Late training batch 15111:

![Train batch 15111](images/improved_train_batch15111.jpg)

Late training batch 15112:

![Train batch 15112](images/improved_train_batch15112.jpg)

## Recommended figures for the report
- Use `images/improved_results.png` to show that the improved run trained successfully and improved steadily.
- Use `images/improved_BoxPR_curve.png` to replace or complement the baseline PR curve.
- Use `images/improved_confusion_matrix_normalized.png` to show that background misses remain the main error pattern.
- Use one prediction/label pair, preferably `images/improved_val_batch0_pred.jpg` and `images/improved_val_batch0_labels.jpg`, for visual examples.

## Recommended figures for the presentation
- Use `images/improved_results.png` for the training slide.
- Use `images/improved_BoxPR_curve.png` or `images/improved_confusion_matrix_normalized.png` for the comparison slide.
- Use `images/improved_val_batch0_pred.jpg` beside `images/improved_val_batch0_labels.jpg` for the qualitative examples slide.
