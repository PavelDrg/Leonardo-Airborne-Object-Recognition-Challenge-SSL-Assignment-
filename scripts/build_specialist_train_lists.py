from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List


CLASSES = [
    "Aircraft",
    "Drone",
    "GroundVehicle",
    "Helicopter",
    "Human",
    "Obstacle",
    "Ship",
]
RARE_CLASS_IDS = {0, 1, 3}


def parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO train lists for small/dense specialist experiments")
    parser.add_argument("--processed-root", type=Path, default=Path("processed"), help="Processed YOLO dataset root")
    parser.add_argument("--small-area-threshold", type=float, default=0.0015, help="Normalized box area for small objects")
    parser.add_argument("--tiny-area-threshold", type=float, default=0.0005, help="Normalized box area for tiny objects")
    parser.add_argument("--large-area-threshold", type=float, default=0.01, help="Median normalized box area for large/clear images")
    parser.add_argument("--dense-threshold", type=int, default=10, help="Object count threshold for dense images")
    parser.add_argument("--very-dense-threshold", type=int, default=30, help="Object count threshold for very dense images")
    parser.add_argument("--small-dense-factors", type=parse_int_list, default=parse_int_list("2,3"), help="Repeat factors to generate")
    parser.add_argument("--rare-factor", type=int, default=2, help="Minimum repeat factor for rare-class images")
    parser.add_argument("--primary-factor", type=int, default=2, help="Factor used for processed/data_small_dense.yaml")
    return parser.parse_args()


def load_manifest(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_label_stats(label_path: Path) -> dict:
    class_counts = Counter()
    areas = []
    if label_path.exists():
        for raw_line in label_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            width = float(parts[3])
            height = float(parts[4])
            class_counts[class_id] += 1
            areas.append(width * height)

    return {
        "object_count": len(areas),
        "class_counts": class_counts,
        "areas": areas,
        "min_area": min(areas) if areas else 0.0,
        "median_area": median(areas) if areas else 0.0,
        "max_area": max(areas) if areas else 0.0,
    }


def yolo_yaml(root: Path, train_list: Path) -> str:
    return "\n".join(
        [
            f"path: {root.resolve().as_posix()}",
            f"train: {train_list.resolve().as_posix()}",
            "val: images/val",
            "names:",
        ]
        + [f"  {idx}: {name}" for idx, name in enumerate(CLASSES)]
        + [""]
    )


def write_train_list(path: Path, image_paths: Iterable[Path]) -> None:
    lines = [image_path.resolve().as_posix() for image_path in image_paths]
    path.write_text("\n".join(lines) + "\n")


def summarize_class_counts(records: Iterable[dict], flag: str) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        if not record[flag]:
            continue
        for class_id, value in record["class_counts"].items():
            counts[CLASSES[class_id]] += value
    return dict(counts)


def main() -> None:
    args = parse_args()
    root = args.processed_root.resolve()
    meta_dir = root / "meta"
    labels_dir = root / "labels" / "train"
    images_dir = root / "images" / "train"

    if not root.exists():
        raise FileNotFoundError(f"Missing processed root: {root}")
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(meta_dir / "train_manifest.csv")
    records = []
    for row in manifest_rows:
        image_id = row["ImageId"]
        stats = load_label_stats(labels_dir / f"{image_id}.txt")
        areas = stats["areas"]
        has_tiny = any(area < args.tiny_area_threshold for area in areas)
        has_small = any(area < args.small_area_threshold for area in areas)
        is_dense = stats["object_count"] >= args.dense_threshold
        is_very_dense = stats["object_count"] >= args.very_dense_threshold
        has_rare = any(class_id in RARE_CLASS_IDS for class_id in stats["class_counts"])
        is_large_clear = stats["object_count"] > 0 and stats["median_area"] >= args.large_area_threshold and not is_dense
        is_small_dense = has_small or is_dense
        records.append(
            {
                "image_id": image_id,
                "object_count": stats["object_count"],
                "class_counts": stats["class_counts"],
                "min_area": stats["min_area"],
                "median_area": stats["median_area"],
                "max_area": stats["max_area"],
                "has_tiny": has_tiny,
                "has_small": has_small,
                "is_dense": is_dense,
                "is_very_dense": is_very_dense,
                "has_rare": has_rare,
                "is_large_clear": is_large_clear,
                "is_small_dense": is_small_dense,
                "image_path": images_dir / f"{image_id}.png",
            }
        )

    for factor in args.small_dense_factors:
        train_paths = []
        for record in records:
            repeats = 1
            if record["is_small_dense"]:
                repeats = max(repeats, factor)
            if record["has_rare"]:
                repeats = max(repeats, args.rare_factor)
            train_paths.extend([record["image_path"]] * repeats)

        list_path = meta_dir / f"train_small_dense_x{factor}_rare_x{args.rare_factor}.txt"
        write_train_list(list_path, train_paths)
        yaml_path = root / f"data_small_dense_x{factor}.yaml"
        yaml_path.write_text(yolo_yaml(root, list_path))
        if factor == args.primary_factor:
            (root / "data_small_dense.yaml").write_text(yolo_yaml(root, list_path))

    large_clear_list = meta_dir / "train_large_clear.txt"
    write_train_list(large_clear_list, [record["image_path"] for record in records if record["is_large_clear"]])
    (root / "data_large_clear.yaml").write_text(yolo_yaml(root, large_clear_list))

    stats_csv = meta_dir / "specialist_image_stats.csv"
    with stats_csv.open("w", newline="") as f:
        fieldnames = [
            "ImageId",
            "object_count",
            "min_area",
            "median_area",
            "max_area",
            "has_tiny",
            "has_small",
            "is_dense",
            "is_very_dense",
            "has_rare",
            "is_large_clear",
            "is_small_dense",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "ImageId": record["image_id"],
                    "object_count": record["object_count"],
                    "min_area": record["min_area"],
                    "median_area": record["median_area"],
                    "max_area": record["max_area"],
                    "has_tiny": int(record["has_tiny"]),
                    "has_small": int(record["has_small"]),
                    "is_dense": int(record["is_dense"]),
                    "is_very_dense": int(record["is_very_dense"]),
                    "has_rare": int(record["has_rare"]),
                    "is_large_clear": int(record["is_large_clear"]),
                    "is_small_dense": int(record["is_small_dense"]),
                }
            )

    summary = {
        "processed_root": str(root),
        "train_images": len(records),
        "small_area_threshold": args.small_area_threshold,
        "tiny_area_threshold": args.tiny_area_threshold,
        "large_area_threshold": args.large_area_threshold,
        "dense_threshold": args.dense_threshold,
        "very_dense_threshold": args.very_dense_threshold,
        "rare_factor": args.rare_factor,
        "small_dense_factors": args.small_dense_factors,
        "counts": {
            "tiny_images": sum(record["has_tiny"] for record in records),
            "small_images": sum(record["has_small"] for record in records),
            "dense_images": sum(record["is_dense"] for record in records),
            "very_dense_images": sum(record["is_very_dense"] for record in records),
            "rare_images": sum(record["has_rare"] for record in records),
            "small_dense_images": sum(record["is_small_dense"] for record in records),
            "large_clear_images": sum(record["is_large_clear"] for record in records),
        },
        "small_dense_class_counts": summarize_class_counts(records, "is_small_dense"),
        "large_clear_class_counts": summarize_class_counts(records, "is_large_clear"),
        "outputs": {
            "stats_csv": str(stats_csv),
            "primary_yaml": str(root / "data_small_dense.yaml"),
            "large_clear_yaml": str(root / "data_large_clear.yaml"),
        },
    }
    summary_path = meta_dir / "specialist_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
