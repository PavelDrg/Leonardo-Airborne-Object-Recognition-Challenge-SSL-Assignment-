from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO on Leonardo object detection data")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed data.yaml")
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/val"), help="Where evaluation artifacts are saved")
    parser.add_argument("--name", default=None, help="Run name")
    return parser.parse_args()


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

    run_name = args.name or args.weights.stem
    model = YOLO(str(args.weights.resolve()))
    metrics = model.val(
        data=str(args.data),
        device=args.device,
        project=str(args.project.resolve()),
        name=run_name,
        plots=True,
        save_json=True,
        save_txt=True,
    )

    out_dir = args.project.resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "box_precision": float(getattr(metrics.box, "mp", 0.0)),
        "box_recall": float(getattr(metrics.box, "mr", 0.0)),
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "map50_95": float(getattr(metrics.box, "map", 0.0)),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
