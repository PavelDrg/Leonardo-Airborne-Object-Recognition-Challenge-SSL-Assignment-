from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


def parse_float_list(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune YOLO validation confidence, NMS IoU, and max_det settings")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained YOLO weights")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed YOLO data.yaml")
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/threshold_tuning"), help="Where tuning artifacts are saved")
    parser.add_argument("--name", default="improved_threshold_grid", help="Subfolder name for this tuning run")
    parser.add_argument(
        "--conf-values",
        type=parse_float_list,
        default=parse_float_list("0.001,0.01,0.05,0.10,0.15,0.20,0.25"),
        help="Comma-separated confidence thresholds",
    )
    parser.add_argument(
        "--iou-values",
        type=parse_float_list,
        default=parse_float_list("0.45,0.55,0.65,0.70,0.75"),
        help="Comma-separated NMS IoU thresholds",
    )
    parser.add_argument(
        "--max-det-values",
        type=parse_int_list,
        default=parse_int_list("300,600,1000"),
        help="Comma-separated max_det values",
    )
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics plots for each validation setting")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style prediction JSON for each setting")
    return parser.parse_args()


def metric_row(metrics, conf: float, iou: float, max_det: int, run_name: str) -> dict:
    return {
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "map50_95": float(getattr(metrics.box, "map", 0.0)),
        "run_name": run_name,
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fieldnames = ["conf", "iou", "max_det", "precision", "recall", "map50", "map50_95", "run_name"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    weights = args.weights.resolve()
    data = args.data.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Missing data YAML: {data}")

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] CUDA available: {torch.cuda.is_available()} | using: {device_name}", flush=True)

    out_dir = args.project.resolve() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    rows = []

    for conf in args.conf_values:
        for iou in args.iou_values:
            for max_det in args.max_det_values:
                run_name = f"conf_{conf:g}_iou_{iou:g}_maxdet_{max_det}".replace(".", "p")
                print(f"Evaluating conf={conf:g}, iou={iou:g}, max_det={max_det} ...", flush=True)
                metrics = model.val(
                    data=str(data),
                    device=args.device,
                    project=str(out_dir),
                    name=run_name,
                    exist_ok=True,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    plots=args.plots,
                    save_json=args.save_json,
                    save_txt=False,
                    verbose=False,
                )
                row = metric_row(metrics, conf, iou, max_det, run_name)
                rows.append(row)
                write_csv(out_dir / "threshold_grid.csv", rows)

    best_by_map = max(rows, key=lambda row: row["map50_95"])
    best_by_recall = max(rows, key=lambda row: row["recall"])
    summary = {
        "weights": str(weights),
        "data": str(data),
        "best_by_map50_95": best_by_map,
        "best_by_recall": best_by_recall,
        "rows": len(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
