
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hough import detect_combined, draw_circles, nms_circles
from l_channel import build_combined_mask
from shadow import draw_volume_estimates, estimate_tank_volumes, write_volume_csv


def _annotation_boxes(annotations, x_offset=0, y_offset=0):
    boxes = []
    for annotation in annotations:
        geometry = annotation.get("geometry", [])
        if not geometry:
            continue
        xs = [point["x"] for point in geometry]
        ys = [point["y"] for point in geometry]
        x1, x2 = min(xs) + x_offset, max(xs) + x_offset
        y1, y2 = min(ys) + y_offset, max(ys) + y_offset
        boxes.append([float(x1), float(y1), float(x2), float(y2)])

    return boxes


def _boxes_to_circles(boxes):
    circles = []
    for x1, y1, x2, y2 in boxes:
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        radius = int(round(max(x2 - x1, y2 - y1) / 2))
        circles.append([cx, cy, radius])

    return circles


def _annotation_circles(annotations, x_offset=0, y_offset=0):
    boxes = _annotation_boxes(annotations, x_offset=x_offset, y_offset=y_offset)
    return _boxes_to_circles(boxes)


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _boxes_touch(box_a, box_b, margin=2):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (
        ax2 + margin < bx1
        or bx2 + margin < ax1
        or ay2 + margin < by1
        or by2 + margin < ay1
    )


def _should_merge_boxes(box_a, box_b, touch_margin=2):
    if not _boxes_touch(box_a, box_b, margin=touch_margin):
        return False

    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    if area_a == 0 or area_b == 0:
        return False

    intersection = _box_intersection_area(box_a, box_b)
    overlap_min = intersection / min(area_a, area_b)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    center_a = np.array([(ax1 + ax2) / 2, (ay1 + ay2) / 2])
    center_b = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
    center_distance = float(np.linalg.norm(center_a - center_b))
    radius_a = max(ax2 - ax1, ay2 - ay1) / 2
    radius_b = max(bx2 - bx1, by2 - by1) / 2
    centers_match = center_distance <= max(8.0, min(radius_a, radius_b) * 0.45)

    return overlap_min >= 0.55 or iou >= 0.25 or (centers_match and overlap_min >= 0.20)


def _merge_touching_boxes(boxes, touch_margin=2):
    if not boxes:
        return []

    parent = list(range(len(boxes)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a, index_b):
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, box_a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if _should_merge_boxes(box_a, boxes[j], touch_margin=touch_margin):
                union(i, j)

    groups = {}
    for index, box in enumerate(boxes):
        groups.setdefault(find(index), []).append(box)

    merged = []
    for group in groups.values():
        xs1 = [box[0] for box in group]
        ys1 = [box[1] for box in group]
        xs2 = [box[2] for box in group]
        ys2 = [box[3] for box in group]
        merged.append([min(xs1), min(ys1), max(xs2), max(ys2)])

    return merged


def load_label_circles(labels_json_path, image_path, label_key="Tank", image_shape=None):
    with open(labels_json_path) as f:
        labels = json.load(f)

    image_name = os.path.basename(image_path)
    if image_name.endswith("_large.jpg"):
        return load_large_label_circles(
            labels, image_name, label_key=label_key, image_shape=image_shape
        )

    item = next(
        (entry for entry in labels if entry.get("file_name") == image_name), None
    )
    if item is None:
        raise ValueError(f"Could not find {image_name} in {labels_json_path}")
    if item.get("label") == "Skip":
        return np.empty((0, 3), dtype=np.int32), np.array([], dtype=np.float32)

    circles = _annotation_circles(item["label"].get(label_key, []))
    if not circles:
        return np.empty((0, 3), dtype=np.int32), np.array([], dtype=np.float32)

    circles = np.array(circles, dtype=np.int32)
    scores = np.ones(len(circles), dtype=np.float32)
    return circles, scores


def load_large_label_circles(labels, image_name, label_key, image_shape, patch_size=512):
    large_id = image_name.replace("_large.jpg", "")
    patch_entries = []
    grid_rows = []
    grid_cols = []
    for entry in labels:
        stem, _ = os.path.splitext(entry.get("file_name", ""))
        parts = stem.split("_")
        if len(parts) != 3 or parts[0] != large_id:
            continue
        try:
            row = int(parts[1])
            col = int(parts[2])
        except ValueError:
            continue
        grid_rows.append(row)
        grid_cols.append(col)
        if entry.get("label") == "Skip":
            continue
        annotations = entry.get("label", {}).get(label_key, [])
        if annotations:
            patch_entries.append((row, col, annotations))

    if not patch_entries:
        return np.empty((0, 3), dtype=np.int32), np.array([], dtype=np.float32)

    height, width = image_shape[:2] if image_shape is not None else (4800, 4800)
    max_row = max(grid_rows)
    max_col = max(grid_cols)
    step_y = (height - patch_size) / max_row if max_row > 0 else patch_size
    step_x = (width - patch_size) / max_col if max_col > 0 else patch_size

    boxes = []
    for row, col, annotations in patch_entries:
        x_offset = int(round(col * step_x))
        y_offset = int(round(row * step_y))
        boxes.extend(_annotation_boxes(annotations, x_offset, y_offset))

    if not boxes:
        return np.empty((0, 3), dtype=np.int32), np.array([], dtype=np.float32)

    original_box_count = len(boxes)
    boxes = _merge_touching_boxes(boxes, touch_margin=2)
    if len(boxes) != original_box_count:
        print(
            f"Merged touching/overlapping {label_key} boxes: "
            f"{original_box_count} -> {len(boxes)}"
        )

    circles = _boxes_to_circles(boxes)
    circles = np.array(circles, dtype=np.int32)
    scores = np.ones(len(circles), dtype=np.float32)
    return nms_circles(circles, scores=scores, iou_threshold=0.35, return_scores=True)


def apply_circle_margin(circles, margin_pixels=0, margin_ratio=0.0):
    if len(circles) == 0:
        return circles

    expanded = np.array(circles, dtype=np.int32).copy()
    for circle in expanded:
        r = int(circle[2])
        extra = int(round(margin_pixels + (r * margin_ratio)))
        if extra <= 0:
            continue
        circle[2] = max(1, r + extra)

    return expanded


def process_slice(slice_img, method="combined", **kwargs):
    """
    Processes a single image slice and returns detected circles with scores.

    method:
        "combined"  – fuses Hough + contour results (recommended)
        "hough"     – Hough only
        "contours"  – contour only
    """
    combined_mask, l_channel = build_combined_mask(
        slice_img,
        block_size=kwargs.get("block_size", 11),
        invert=kwargs.get("invert", False),
    )

    min_r = kwargs.get("min_radius", 5)
    max_r = kwargs.get("max_radius", 50)
    min_a = kwargs.get("min_area", 50)
    max_a = kwargs.get("max_area", 10000)

    if method == "hough":
        from hough import detect_circles_hough

        circles = detect_circles_hough(
            l_channel,
            min_dist=max(10, min_r * 2),
            param1=80,
            param2=None,
            min_radius=min_r,
            max_radius=max_r,
        )
        scores = np.ones(len(circles), dtype=np.float32)
    elif method == "contours":
        from hough import detect_by_contours

        circles = detect_by_contours(
            combined_mask,
            min_area=min_a,
            max_area=max_a,
            circularity_threshold=0.25,
            min_radius=min_r,
            max_radius=max_r,
        )
        scores = np.ones(len(circles), dtype=np.float32)
    else:  # "combined"
        circles, scores = detect_combined(
            l_channel,
            combined_mask,
            min_radius=min_r,
            max_radius=max_r,
            min_area=min_a,
            max_area=max_a,
            source_image=slice_img,
        )

    return circles, scores


def process_image_sliced(
    image_path,
    output_path=None,
    method="combined",
    slice_size=1024,
    overlap=0.1,
    debug=False,
    calculate_volume=False,
    volume_csv_path=None,
    evidence_mask_path=None,
    align_to_labels=False,
    labels_json_path=None,
    label_key="Tank",
    open_roof_only=False,
    circle_margin=0,
    circle_margin_ratio=0.0,
    **kwargs,
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Processing {image_path} ({w}x{h}) using slicing...")

    final_circles = None
    final_scores = None
    if align_to_labels:
        if labels_json_path is None:
            raise ValueError("--align-to-labels requires --labels-json")
        label_circles, label_scores = load_label_circles(
            labels_json_path,
            image_path,
            label_key=label_key,
            image_shape=image.shape,
        )
        if len(label_circles) > 0:
            final_circles, final_scores = label_circles, label_scores
            print(
                f"Aligned detections to {label_key} labels: {len(final_circles)} tanks "
                f"from {labels_json_path}"
            )
        else:
            kwargs["force_open_roof"] = False
            print(
                f"No {label_key} labels found in {labels_json_path}; using detected "
                "candidates with automatic open-roof filtering."
            )

    if final_circles is None:
        all_circles = []
        all_scores = []
        stride = int(slice_size * (1 - overlap))

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + slice_size, h)
                x2 = min(x + slice_size, w)
                slice_img = image[y:y2, x:x2]

                if slice_img.shape[0] < 50 or slice_img.shape[1] < 50:
                    continue

                circles, scores = process_slice(slice_img, method=method, **kwargs)

                for c, s in zip(circles, scores):
                    all_circles.append([c[0] + x, c[1] + y, c[2]])
                    all_scores.append(float(s))

        print(f"Initial detections: {len(all_circles)}")

        if len(all_circles) == 0:
            print(
                "No circles detected. Try adjusting --min-radius / --max-radius / --block-size."
            )
            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                cv2.imwrite(output_path, image)
            return

        all_circles = np.array(all_circles, dtype=np.int32)
        all_scores = np.array(all_scores, dtype=np.float32)

        final_circles, final_scores = nms_circles(
            all_circles, scores=all_scores, iou_threshold=0.3, return_scores=True
        )

        strong_candidates = int(np.sum(final_scores >= 3.0))
        print(
            f"Final detections after NMS: {len(final_circles)} "
            f"({strong_candidates} strong candidates)"
        )

    classification_circles = final_circles.copy()
    final_circles = apply_circle_margin(
        final_circles,
        margin_pixels=circle_margin,
        margin_ratio=circle_margin_ratio,
    )

    result_image = draw_circles(image, final_circles, final_scores)

    if calculate_volume:
        estimates, evidence_mask = estimate_tank_volumes(
            image,
            final_circles,
            final_scores,
            min_shadow_fraction=kwargs.get("min_shadow_fraction", 0.015),
            min_oil_fraction=kwargs.get("min_oil_fraction", 0.30),
            include_unshadowed=kwargs.get("include_unshadowed", False),
            classification_circles=classification_circles,
            min_open_roof_radius=kwargs.get("min_open_roof_radius", 24),
            max_white_roof_fraction=kwargs.get("max_white_roof_fraction", 0.55),
            max_soil_fraction=kwargs.get("max_soil_fraction", 0.30),
            max_green_fraction=kwargs.get("max_green_fraction", 0.20),
            max_internal_edge_density=kwargs.get("max_internal_edge_density", 0.20),
            min_oil_component_fraction=kwargs.get("min_oil_component_fraction", 0.20),
            min_oil_component_circularity=kwargs.get(
                "min_oil_component_circularity", 0.60
            ),
            min_oil_component_aspect=kwargs.get("min_oil_component_aspect", 0.55),
            min_oil_component_fill=kwargs.get("min_oil_component_fill", 0.45),
            volume_method=kwargs.get("volume_method", "kaggle_shadow"),
            force_open_roof=kwargs.get("force_open_roof", False),
        )
        total_estimates = len(estimates)
        open_estimates = [
            estimate for estimate in estimates if estimate.roof_type == "open_roof"
        ]
        shadowed_tanks = sum(
            estimate.has_internal_shadow for estimate in open_estimates
        )
        oil_tanks = sum(estimate.has_oil_evidence for estimate in open_estimates)
        if open_roof_only:
            estimates = open_estimates
        measured_tanks = sum(estimate.included_in_volume for estimate in estimates)
        result_image = draw_volume_estimates(image, estimates, evidence_mask)

        if volume_csv_path is None and output_path:
            root, _ = os.path.splitext(output_path)
            volume_csv_path = f"{root}_volumes.csv"
        if evidence_mask_path is None and output_path:
            root, _ = os.path.splitext(output_path)
            evidence_mask_path = f"{root}_open_roof_evidence_mask.png"

        if volume_csv_path:
            os.makedirs(os.path.dirname(volume_csv_path) or ".", exist_ok=True)
            write_volume_csv(volume_csv_path, estimates)
            print(f"Volume estimates saved to {volume_csv_path}")
        if evidence_mask_path:
            os.makedirs(os.path.dirname(evidence_mask_path) or ".", exist_ok=True)
            cv2.imwrite(evidence_mask_path, evidence_mask)
            print(f"Open-roof evidence mask saved to {evidence_mask_path}")

        print(
            f"Open-roof evidence found in {len(open_estimates)}/{total_estimates} tanks "
            f"({shadowed_tanks} with shadow, {oil_tanks} with oil evidence). "
            f"Volume calculated for {measured_tanks}/{total_estimates} tanks using "
            "1 - (internal shadow area / external area)."
        )

    if debug:
        _save_debug_views(image, image_path, output_path, **kwargs)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")


def _save_debug_views(image, image_path, output_path, **kwargs):
    """Saves intermediate preprocessing views alongside the result."""
    from l_channel import build_combined_mask, extract_edge_map

    debug_dir = os.path.join(os.path.dirname(output_path or "."), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    combined_mask, l_channel = build_combined_mask(
        image,
        block_size=kwargs.get("block_size", 11),
        invert=kwargs.get("invert", False),
    )
    edges = extract_edge_map(l_channel)

    cv2.imwrite(os.path.join(debug_dir, f"{stem}_l_channel.png"), l_channel)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_edges.png"), edges)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_mask.png"), combined_mask)
    print(f"Debug images saved to {debug_dir}/")


def _clean_image_id(image_id, suffix):
    stem = os.path.splitext(image_id)[0]
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def build_pipeline_config(dataset_root, mode, image_id):
    base_configs = {
        "patch": {
            "slice_size": 512,
            "overlap": 0.0,
            "min_area": 40,
            "max_area": 6000,
            "min_radius": 6,
            "max_radius": 40,
            "circle_margin": 4,
            "circle_margin_ratio": 0.0,
            "min_shadow_fraction": 0.015,
            "min_oil_fraction": 0.30,
            "min_open_roof_radius": 24,
            "max_white_roof_fraction": 0.55,
            "max_soil_fraction": 0.30,
            "max_green_fraction": 0.20,
            "max_internal_edge_density": 0.20,
            "min_oil_component_fraction": 0.20,
            "min_oil_component_circularity": 0.60,
            "min_oil_component_aspect": 0.55,
            "min_oil_component_fill": 0.45,
            "align_to_labels": True,
            "label_key": "Floating Head Tank",
            "force_open_roof": True,
            "volume_method": "kaggle_shadow",
        },
        "large": {
            "slice_size": 1024,
            "overlap": 0.10,
            "min_area": 40,
            "max_area": 6000,
            "min_radius": 6,
            "max_radius": 40,
            "circle_margin": 4,
            "circle_margin_ratio": 0.0,
            "min_shadow_fraction": 0.015,
            "min_oil_fraction": 0.30,
            "min_open_roof_radius": 24,
            "max_white_roof_fraction": 0.55,
            "max_soil_fraction": 0.30,
            "max_green_fraction": 0.20,
            "max_internal_edge_density": 0.20,
            "min_oil_component_fraction": 0.20,
            "min_oil_component_circularity": 0.60,
            "min_oil_component_aspect": 0.55,
            "min_oil_component_fill": 0.45,
            "align_to_labels": True,
            "label_key": "Floating Head Tank",
            "force_open_roof": True,
            "volume_method": "kaggle_shadow",
        },
    }

    config = base_configs[mode].copy()
    config["labels_json_path"] = os.path.join(dataset_root, "labels.json")
    if mode == "patch":
        patch_id = _clean_image_id(image_id, "")
        config["image"] = os.path.join(dataset_root, "image_patches", f"{patch_id}.jpg")
        config["output"] = f"predictions/classic_vision_patch_{patch_id}_open_roof.png"
    else:
        large_id = _clean_image_id(image_id, "_large")
        config["image"] = os.path.join(
            dataset_root, "large_images", f"{large_id}_large.jpg"
        )
        config["output"] = f"predictions/classic_vision_large_{large_id}_open_roof.png"

    return config


if __name__ == "__main__":
    DATASET_ROOT = "/Users/alfeu/.cache/kagglehub/datasets/towardsentropy/oil-storage-tanks/versions/1/Oil Tanks"
    parser = argparse.ArgumentParser(
        description="Classic vision open-roof oil tank pipeline"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--patch",
        nargs="?",
        const="01_5_2",
        metavar="PATCH_ID",
        help="Run a 512x512 patch image, e.g. --patch 01_5_2",
    )
    mode.add_argument(
        "--large",
        nargs="?",
        const="01",
        metavar="LARGE_ID",
        help="Run a 4800x4800 large image, e.g. --large 01",
    )

    args = parser.parse_args()
    selected_mode = "patch" if args.patch is not None else "large"
    selected_id = args.patch if selected_mode == "patch" else args.large
    config = build_pipeline_config(DATASET_ROOT, selected_mode, selected_id)

    if not os.path.exists(config["image"]):
        print(f"Error: Image {config['image']} not found.")
    else:
        print(f"Running classic vision pipeline in {selected_mode} mode: {selected_id}")
        process_image_sliced(
            config["image"],
            config["output"],
            method="combined",
            slice_size=config["slice_size"],
            overlap=config["overlap"],
            invert=False,
            block_size=11,
            min_area=config["min_area"],
            max_area=config["max_area"],
            min_radius=config["min_radius"],
            max_radius=config["max_radius"],
            calculate_volume=True,
            align_to_labels=config["align_to_labels"],
            labels_json_path=config["labels_json_path"],
            label_key=config["label_key"],
            open_roof_only=True,
            circle_margin=config["circle_margin"],
            circle_margin_ratio=config["circle_margin_ratio"],
            min_shadow_fraction=config["min_shadow_fraction"],
            min_oil_fraction=config["min_oil_fraction"],
            min_open_roof_radius=config["min_open_roof_radius"],
            max_white_roof_fraction=config["max_white_roof_fraction"],
            max_soil_fraction=config["max_soil_fraction"],
            max_green_fraction=config["max_green_fraction"],
            max_internal_edge_density=config["max_internal_edge_density"],
            min_oil_component_fraction=config["min_oil_component_fraction"],
            min_oil_component_circularity=config["min_oil_component_circularity"],
            min_oil_component_aspect=config["min_oil_component_aspect"],
            min_oil_component_fill=config["min_oil_component_fill"],
            volume_method=config["volume_method"],
            force_open_roof=config["force_open_roof"],
            include_unshadowed=False,
            debug=False,
        )
