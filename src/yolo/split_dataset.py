import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def yolo_line_from_coco_bbox(
    *, bbox_xywh: list[float], img_w: int, img_h: int, class_id: int
) -> str:
    x, y, w, h = bbox_xywh
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{class_id} {x_center:.10f} {y_center:.10f} {w_norm:.10f} {h_norm:.10f}"


@dataclass(frozen=True)
class Split:
    train_groups: set[str]
    val_groups: set[str]


def group_from_filename(filename: str) -> str:
    # Example: "01_0_0.jpg" -> "01" (origin large image)
    return filename.split("_")[0]


def make_split(groups: Iterable[str], *, val_ratio: float, seed: int) -> Split:
    groups = sorted(set(groups))
    if not groups:
        raise ValueError("No groups found. Check your images directory.")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    rng = random.Random(seed)
    rng.shuffle(groups)

    val_n = max(1, int(round(len(groups) * val_ratio)))
    val_groups = set(groups[:val_n])
    train_groups = set(groups[val_n:])

    if not train_groups:
        raise ValueError(
            "Train split ended up empty. Decrease val_ratio or increase dataset size."
        )

    return Split(train_groups=train_groups, val_groups=val_groups)


def safe_symlink_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if copy:
        dst.write_bytes(src.read_bytes())
        return

    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        # Fallback for filesystems that don't allow symlinks
        dst.write_bytes(src.read_bytes())


def write_dataset_yaml(out_root: Path) -> None:
    # Ultralytics expects:
    # path: <root>
    # train: images/train
    # val: images/val
    yaml = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: floating_head_tank\n"
    )
    (out_root / "oil_tanks.yaml").write_text(yaml)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Oil Storage Tanks dataset into YOLO train/val folders (grouped by large image id)."
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=Path("dataset/oil_tanks/fixed_coco/labels_coco_fixed.json"),
        help="Path to COCO json with width/height (default: dataset/oil_tanks/fixed_coco/labels_coco_fixed.json)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/oil_tanks/image_patches"),
        help="Directory with patch images (default: dataset/oil_tanks/image_patches)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/yolo"),
        help="Output directory (default: dataset/yolo)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of groups to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle groups (default: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (slower, uses more disk).",
    )

    args = parser.parse_args()

    coco_path: Path = args.coco
    images_dir: Path = args.images_dir
    out_root: Path = args.out

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    coco = json.loads(coco_path.read_text())
    images = coco["images"]
    annotations = coco["annotations"]

    images_by_id = {img["id"]: img for img in images}

    # Gather groups from filenames
    groups = [group_from_filename(img["file_name"]) for img in images]
    split = make_split(groups, val_ratio=args.val_ratio, seed=args.seed)

    ann_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        ann_by_image_id[int(ann["image_id"])].append(ann)

    # Prepare output dirs
    for sub in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Create files
    for img in images:
        file_name = img["file_name"]
        group = group_from_filename(file_name)

        if group in split.val_groups:
            split_name = "val"
        elif group in split.train_groups:
            split_name = "train"
        else:
            # Shouldn't happen
            continue

        src_img = images_dir / file_name
        if not src_img.exists():
            raise FileNotFoundError(f"Image referenced in COCO not found: {src_img}")

        dst_img = out_root / "images" / split_name / file_name
        safe_symlink_or_copy(src_img, dst_img, copy=args.copy)

        # Write YOLO label file (empty file for background images)
        stem = Path(file_name).stem
        dst_label = out_root / "labels" / split_name / f"{stem}.txt"

        img_w = int(img.get("width", 0))
        img_h = int(img.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            raise ValueError(
                f"Missing/invalid width/height for image {file_name}. "
                "Use the fixed COCO json with width/height."
            )

        yolo_lines: list[str] = []
        for ann in ann_by_image_id.get(int(img["id"]), []):
            # COCO category_id starts at 1; map to YOLO class 0
            class_id = int(ann["category_id"]) - 1
            yolo_lines.append(
                yolo_line_from_coco_bbox(
                    bbox_xywh=ann["bbox"],
                    img_w=img_w,
                    img_h=img_h,
                    class_id=class_id,
                )
            )

        dst_label.write_text("\n".join(yolo_lines))

    # Save split info + dataset yaml
    splits_json = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_groups": sorted(split.train_groups),
        "val_groups": sorted(split.val_groups),
    }
    (out_root / "splits.json").write_text(json.dumps(splits_json, indent=2))
    write_dataset_yaml(out_root)

    print(f"Done. YOLO dataset written to: {out_root}")
    print(f"YAML: {out_root / 'oil_tanks.yaml'}")
    print(f"Splits: {out_root / 'splits.json'}")


if __name__ == "__main__":
    main()
