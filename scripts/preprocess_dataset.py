from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLASSES = [
    "Aircraft",
    "Drone",
    "GroundVehicle",
    "Helicopter",
    "Human",
    "Obstacle",
    "Ship",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    classes: Tuple[str, ...]
    object_count: int
    has_rare: bool
    is_background: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Leonardo object detection data")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder containing train/ test/ train.csv")
    parser.add_argument("--output-root", type=Path, default=Path("processed"), help="Where to write processed data")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of using symlinks")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output folder")
    return parser.parse_args()


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        _ = struct.unpack(">I", f.read(4))[0]
        if f.read(4) != b"IHDR":
            raise ValueError(f"PNG missing IHDR chunk: {path}")
        w, h = struct.unpack(">II", f.read(8))
        return int(w), int(h)


def load_annotations(
    csv_path: Path,
) -> Tuple[Dict[str, List[Tuple[str, float, float, float, float]]], Dict[str, int], Dict[str, int]]:
    per_image: Dict[str, List[Tuple[str, float, float, float, float]]] = defaultdict(list)
    class_counts = Counter()
    stats = Counter()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["raw_boxes"] += 1
            image_id = row["ImageId"]
            cls = row["class"]
            x1_raw, y1_raw, x2_raw, y2_raw = map(float, row["bbox"].split())
            x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if (x1, y1, x2, y2) != (x1_raw, y1_raw, x2_raw, y2_raw):
                stats["clipped_boxes"] += 1
            if x2 <= x1 or y2 <= y1:
                stats["dropped_boxes"] += 1
                continue
            per_image[image_id].append((cls, x1, y1, x2, y2))
            class_counts[cls] += 1

    return per_image, dict(class_counts), dict(stats)


def build_records(all_image_ids: Sequence[str], per_image: Dict[str, List[Tuple[str, float, float, float, float]]]) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    rare = {"Aircraft", "Drone", "Helicopter"}
    for image_id in all_image_ids:
        boxes = per_image.get(image_id, [])
        classes = tuple(sorted({cls for cls, *_ in boxes}))
        object_count = len(boxes)
        has_rare = any(cls in rare for cls in classes)
        is_background = object_count == 0
        records.append(ImageRecord(image_id, classes, object_count, has_rare, is_background))
    return records


def stratified_split(records: Sequence[ImageRecord], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for rec in records:
        if rec.is_background:
            count_bucket = "background"
        elif rec.object_count == 1:
            count_bucket = "1"
        elif rec.object_count <= 3:
            count_bucket = "2-3"
        elif rec.object_count <= 10:
            count_bucket = "4-10"
        else:
            count_bucket = "11+"
        rare_bucket = "rare" if rec.has_rare else "common"
        buckets[(count_bucket, rare_bucket)].append(rec.image_id)

    train_ids: List[str] = []
    val_ids: List[str] = []
    for ids in buckets.values():
        rng.shuffle(ids)
        n_val = max(1 if len(ids) > 1 else 0, int(round(len(ids) * val_ratio)))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def make_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_labels(
    split_name: str,
    image_ids: Iterable[str],
    per_image: Dict[str, List[Tuple[str, float, float, float, float]]],
    image_src: Path,
    output_root: Path,
    copy_images: bool,
) -> int:
    num_empty = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(list(image_ids), desc=f"Writing {split_name}", unit="img")
    except Exception:
        iterator = image_ids

    for image_id in iterator:
        src_img = image_src / f"{image_id}.png"
        dst_img = output_root / "images" / split_name / f"{image_id}.png"
        link_or_copy(src_img, dst_img, copy_images)

        boxes = per_image.get(image_id, [])
        if not boxes:
            num_empty += 1
        lines = []
        for cls, x1, y1, x2, y2 in boxes:
            class_id = CLASS_TO_ID[cls]
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (output_root / "labels" / split_name / f"{image_id}.txt").write_text("\n".join(lines))
    return num_empty


def image_stats_from_records(records: Sequence[ImageRecord], per_image: Dict[str, List[Tuple[str, float, float, float, float]]]) -> Dict[str, object]:
    counts = Counter(rec.object_count for rec in records)
    rare = sum(1 for rec in records if rec.has_rare)
    background = sum(1 for rec in records if rec.is_background)
    class_counts = Counter()
    for rec in records:
        for cls, *_ in per_image.get(rec.image_id, []):
            class_counts[cls] += 1
    return {
        "images": len(records),
        "background_images": background,
        "rare_images": rare,
        "class_counts": dict(class_counts),
        "object_count_histogram": {str(k): v for k, v in sorted(counts.items())},
    }


def dataset_summary(
    data_root: Path,
    image_ids: Sequence[str],
    per_image: Dict[str, List[Tuple[str, float, float, float, float]]],
    class_counts: Dict[str, int],
    load_stats: Dict[str, int],
    output_root: Path,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> None:
    all_records = build_records(image_ids, per_image)
    by_id = {rec.image_id: rec for rec in all_records}

    out_of_range = 0
    width_values = []
    height_values = []
    area_values = []
    for boxes in per_image.values():
        for _, x1, y1, x2, y2 in boxes:
            if x1 == 0.0 or y1 == 0.0 or x2 == 1.0 or y2 == 1.0:
                out_of_range += 1
            w = x2 - x1
            h = y2 - y1
            width_values.append(w)
            height_values.append(h)
            area_values.append(w * h)

    def quantiles(values: Sequence[float]) -> Dict[str, float]:
        arr = sorted(values)
        if not arr:
            return {}
        def q(p: float) -> float:
            idx = int(round(p * (len(arr) - 1)))
            return arr[idx]
        return {
            "min": arr[0],
            "p25": q(0.25),
            "median": q(0.50),
            "mean": sum(arr) / len(arr),
            "p75": q(0.75),
            "max": arr[-1],
        }

    summary = {
        "data_root": str(data_root),
        "images": len(image_ids),
        "annotated_images": sum(1 for rec in all_records if not rec.is_background),
        "unannotated_images": sum(1 for rec in all_records if rec.is_background),
        "annotations": sum(class_counts.values()),
        "class_counts": class_counts,
        "load_stats": load_stats,
        "bbox_stats": {
            "width": quantiles(width_values),
            "height": quantiles(height_values),
            "area": quantiles(area_values),
            "boxes_touching_bounds_after_clipping": out_of_range,
        },
        "split_stats": {
            "train": image_stats_from_records([by_id[i] for i in train_ids], per_image),
            "val": image_stats_from_records([by_id[i] for i in val_ids], per_image),
        },
    }
    (output_root / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))


def write_manifest(output_root: Path, split_name: str, image_ids: Sequence[str], per_image: Dict[str, List[Tuple[str, float, float, float, float]]]) -> None:
    manifest_path = output_root / "meta" / f"{split_name}_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "object_count", "classes", "has_rare", "is_background"])
        for image_id in image_ids:
            boxes = per_image.get(image_id, [])
            classes = sorted({cls for cls, *_ in boxes})
            writer.writerow([
                image_id,
                len(boxes),
                "|".join(classes),
                int(any(cls in {"Aircraft", "Drone", "Helicopter"} for cls in classes)),
                int(len(boxes) == 0),
            ])


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    train_dir = data_root / "train"
    csv_path = data_root / "train.csv"
    if not train_dir.exists() or not csv_path.exists():
        raise FileNotFoundError("Expected train/ and train.csv under --data-root")

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    make_dirs(output_root)

    per_image, class_counts, load_stats = load_annotations(csv_path)
    image_ids = sorted(p.stem for p in train_dir.glob("*.png"))
    records = build_records(image_ids, per_image)
    train_ids, val_ids = stratified_split(records, args.val_ratio, args.seed)

    train_empty = write_yolo_labels("train", train_ids, per_image, train_dir, output_root, args.copy_images)
    val_empty = write_yolo_labels("val", val_ids, per_image, train_dir, output_root, args.copy_images)

    write_manifest(output_root, "train", train_ids, per_image)
    write_manifest(output_root, "val", val_ids, per_image)
    dataset_summary(data_root, image_ids, per_image, class_counts, load_stats, output_root, train_ids, val_ids)

    yolo_yaml = output_root / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output_root}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        + [f"  {idx}: {name}" for idx, name in enumerate(CLASSES)]
        + [""]
    )
    yolo_yaml.write_text(yaml_text)

    print(json.dumps({
        "output_root": str(output_root),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "train_background_images": train_empty,
        "val_background_images": val_empty,
        "classes": CLASSES,
    }, indent=2))


if __name__ == "__main__":
    main()
