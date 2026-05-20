from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on Leonardo object detection data")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed data.yaml")
    parser.add_argument("--variant", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Training output folder")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--oversample-rare", action="store_true", help="Repeat rare-class images in train split")
    parser.add_argument("--oversample-factor", type=float, default=2.0, help="How many times to repeat rare images")
    parser.add_argument("--imgsz", type=int, default=None, help="Override variant image size")
    parser.add_argument("--epochs", type=int, default=None, help="Override variant epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override variant batch size")
    parser.add_argument("--workers", type=int, default=None, help="Override Ultralytics dataloader workers")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    return parser.parse_args()


def config_for_variant(variant: str) -> dict:
    if variant == "baseline":
        return {
            "weights": "yolo11n.pt",
            "imgsz": 960,
            "epochs": 18,
            "batch": 16,
            "mosaic": 0.8,
            "mixup": 0.05,
            "copy_paste": 0.0,
            "max_det": 300,
            "patience": 5,
            "lr0": 0.01,
        }
    return {
        "weights": "yolo11s.pt",
        "imgsz": 1280,
        "epochs": 20,
        "batch": 6,
        "mosaic": 1.0,
        "mixup": 0.10,
        "copy_paste": 0.15,
        "max_det": 600,
        "patience": 7,
        "lr0": 0.003,
        "close_mosaic": 15,
    }


def build_oversampled_train_list(data_yaml: Path, factor: float) -> Path:
    root = data_yaml.parent.resolve()
    manifest = root / "meta" / "train_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {manifest}")

    image_entries = []
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ImageId"]
            has_rare = row["has_rare"] == "1"
            abs_path = (root / "images" / "train" / f"{img_id}.png").resolve().as_posix()
            repeats = int(round(factor)) if has_rare else 1
            repeats = max(1, repeats)
            image_entries.extend([abs_path] * repeats)

    list_path = root / "meta" / f"train_oversampled_x{factor:g}.txt"
    list_path.write_text("\n".join(image_entries) + "\n")
    return list_path


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] CUDA available: {torch.cuda.is_available()} | using: {device_name}", flush=True)

    cfg = config_for_variant(args.variant)
    for key in ("imgsz", "epochs", "batch", "patience"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.workers is not None:
        cfg["workers"] = args.workers

    name = args.name or f"leonardo_{args.variant}"
    model = YOLO(cfg.pop("weights"))

    data_yaml = args.data.resolve()
    if args.oversample_rare:
        train_list = build_oversampled_train_list(data_yaml, args.oversample_factor)
        temp_yaml = data_yaml.parent / f"data_{args.variant}_oversampled.yaml"
        temp_yaml.write_text(
            "\n".join(
                [
                    f"path: {data_yaml.parent.as_posix()}",
                    f"train: {train_list}",
                    "val: images/val",
                    "names:",
                    "  0: Aircraft",
                    "  1: Drone",
                    "  2: GroundVehicle",
                    "  3: Helicopter",
                    "  4: Human",
                    "  5: Obstacle",
                    "  6: Ship",
                    "",
                ]
            )
        )
        data_yaml = temp_yaml

    model.train(
        data=str(data_yaml),
        project=str(args.project.resolve()),
        name=name,
        device=args.device,
        **cfg,
    )

    best_pt = args.project.resolve() / name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"Running validation on {best_pt} ...", flush=True)
        best_model = YOLO(str(best_pt))
        best_model.val(data=str(args.data), device=args.device, plots=True, save_json=True, save_txt=True)


if __name__ == "__main__":
    main()
