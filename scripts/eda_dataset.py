from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


CLASSES = [
    "Aircraft",
    "Drone",
    "GroundVehicle",
    "Helicopter",
    "Human",
    "Obstacle",
    "Ship",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for Leonardo airborne object dataset")
    parser.add_argument("--data-root", type=Path, required=True, help="Folder with train/ and train.csv")
    parser.add_argument("--processed-root", type=Path, default=None, help="Folder created by preprocess_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=Path("Notes/eda"), help="Where plots are written")
    parser.add_argument("--sample-size", type=int, default=300, help="How many images to sample for size plots")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not PNG: {path}")
        _ = struct.unpack(">I", f.read(4))[0]
        if f.read(4) != b"IHDR":
            raise ValueError(f"Malformed PNG: {path}")
        return struct.unpack(">II", f.read(8))


def load_raw_stats(data_root: Path) -> Dict[str, object]:
    train_dir = data_root / "train"
    csv_path = data_root / "train.csv"

    image_ids = sorted(p.stem for p in train_dir.glob("*.png"))
    rows: List[Tuple[str, str, float, float, float, float]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            x1, y1, x2, y2 = map(float, row["bbox"].split())
            rows.append((row["ImageId"], row["class"], x1, y1, x2, y2))

    class_counts = Counter(r[1] for r in rows)
    by_image = defaultdict(int)
    for image_id, *_ in rows:
        by_image[image_id] += 1

    width_vals = []
    height_vals = []
    area_vals = []
    clipped = 0
    for _, _, x1, y1, x2, y2 in rows:
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x1 == 0.0 or y1 == 0.0 or x2 == 1.0 or y2 == 1.0:
            clipped += 1
        w = x2 - x1
        h = y2 - y1
        width_vals.append(w)
        height_vals.append(h)
        area_vals.append(w * h)

    return {
        "images": len(image_ids),
        "annotations": len(rows),
        "class_counts": class_counts,
        "objects_per_image": by_image,
        "width_vals": width_vals,
        "height_vals": height_vals,
        "area_vals": area_vals,
        "clipped_boxes": clipped,
        "image_ids": image_ids,
        "train_dir": train_dir,
    }


def plot_bar(labels: Sequence[str], values: Sequence[int], title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#3b82f6")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.2)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_hist(values: Sequence[float], title: str, path: Path, bins: int = 30, logy: bool = False) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, color="#10b981", edgecolor="white", log=logy)
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def sample_sizes(train_dir: Path, image_ids: Sequence[str], sample_size: int) -> Dict[str, int]:
    import random

    rng = random.Random(42)
    sample = list(image_ids)
    rng.shuffle(sample)
    sample = sample[: min(sample_size, len(sample))]
    sizes = Counter()
    for image_id in sample:
        sizes[str(read_png_size(train_dir / f"{image_id}.png"))] += 1
    return sizes


def load_processed_summary(processed_root: Path) -> Dict[str, object] | None:
    summary_path = processed_root / "meta" / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def plot_split_comparison(summary: Dict[str, object], out_dir: Path) -> None:
    split_stats = summary["split_stats"]
    train_hist = split_stats["train"]["object_count_histogram"]
    val_hist = split_stats["val"]["object_count_histogram"]
    keys = sorted({int(k) for k in train_hist.keys()} | {int(k) for k in val_hist.keys()})
    train_vals = [train_hist.get(str(k), 0) for k in keys]
    val_vals = [val_hist.get(str(k), 0) for k in keys]

    x = range(len(keys))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.2 for i in x], train_vals, width=0.4, label="train", color="#2563eb")
    plt.bar([i + 0.2 for i in x], val_vals, width=0.4, label="val", color="#f97316")
    plt.xticks(list(x), [str(k) for k in keys])
    plt.title("Object count per image after split")
    plt.xlabel("objects per image")
    plt.ylabel("images")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "processed_object_count_split.png", dpi=180)
    plt.close()

    raw_classes = summary["class_counts"]
    train_classes = split_stats["train"]["class_counts"]
    val_classes = split_stats["val"]["class_counts"]
    class_labels = CLASSES
    x = range(len(class_labels))
    train_vals = [train_classes.get(c, 0) for c in class_labels]
    val_vals = [val_classes.get(c, 0) for c in class_labels]
    raw_vals = [raw_classes.get(c, 0) for c in class_labels]
    plt.figure(figsize=(11, 5))
    plt.bar([i - 0.25 for i in x], raw_vals, width=0.25, label="raw", color="#94a3b8")
    plt.bar([i for i in x], train_vals, width=0.25, label="train", color="#2563eb")
    plt.bar([i + 0.25 for i in x], val_vals, width=0.25, label="val", color="#f97316")
    plt.xticks(list(x), class_labels, rotation=25, ha="right")
    plt.ylabel("boxes")
    plt.title("Class distribution before and after split")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution_raw_train_val.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    ensure_dir(out_dir)

    raw = load_raw_stats(data_root)

    plot_bar(list(raw["class_counts"].keys()), list(raw["class_counts"].values()), "Class distribution", out_dir / "raw_class_distribution.png")

    obj_hist = Counter(raw["objects_per_image"].values())
    keys = list(range(0, min(12, max(obj_hist) + 1)))
    values = [obj_hist.get(k, 0) for k in keys]
    plot_bar([str(k) for k in keys], values, "Objects per image", out_dir / "raw_objects_per_image.png")

    plot_hist(raw["width_vals"], "Bounding box width", out_dir / "raw_bbox_width.png")
    plot_hist(raw["height_vals"], "Bounding box height", out_dir / "raw_bbox_height.png")
    plot_hist(raw["area_vals"], "Bounding box area", out_dir / "raw_bbox_area.png", logy=True)

    size_counts = sample_sizes(raw["train_dir"], raw["image_ids"], args.sample_size)
    plot_bar(list(size_counts.keys()), list(size_counts.values()), "Sampled image sizes", out_dir / "raw_image_sizes_sample.png")

    raw_summary = {
        "images": raw["images"],
        "annotations": raw["annotations"],
        "clipped_boxes": raw["clipped_boxes"],
        "class_counts": dict(raw["class_counts"]),
        "objects_per_image_histogram": {str(k): v for k, v in sorted(obj_hist.items())},
    }
    (out_dir / "raw_summary.json").write_text(json.dumps(raw_summary, indent=2))

    if args.processed_root is not None:
        summary = load_processed_summary(args.processed_root.resolve())
        if summary is not None:
            plot_split_comparison(summary, out_dir)
            (out_dir / "processed_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({"output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
