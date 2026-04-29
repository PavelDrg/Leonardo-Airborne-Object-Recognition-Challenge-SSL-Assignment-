# Run Order

1. Generate the raw dataset report.
   - `python3 scripts/eda_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --output-dir "Notes/eda_raw"`

2. Build the processed dataset mirror.
   - `python3 scripts/preprocess_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --output-root "processed" --force`

3. Re-run EDA on the processed split.
   - `python3 scripts/eda_dataset.py --data-root "leonardo-airborne-object-recognition-challenge" --processed-root "processed" --output-dir "Notes/eda_processed"`

4. Train the baseline detector.
   - `python3 scripts/train_yolo.py --data "processed/data.yaml" --variant baseline --project "runs/detect"`

5. Train the stronger detector.
   - `python3 scripts/train_yolo.py --data "processed/data.yaml" --variant improved --oversample-rare --oversample-factor 2 --project "runs/detect"`

6. Evaluate the best checkpoint explicitly if needed.
   - `python3 scripts/evaluate_yolo.py --weights "runs/detect/leonardo_baseline/weights/best.pt" --data "processed/data.yaml" --project "runs/val"`

7. Compare the training results and screenshots in the report.
