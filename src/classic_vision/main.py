import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hough import detect_combined, draw_circles, nms_circles
from l_channel import build_combined_mask
from shadow import draw_volume_estimates, estimate_tank_volumes, write_volume_csv


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
    **kwargs,
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Processing {image_path} ({w}x{h}) using slicing...")

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

    result_image = draw_circles(image, final_circles, final_scores)

    if calculate_volume:
        estimates, evidence_mask = estimate_tank_volumes(
            image,
            final_circles,
            final_scores,
            min_shadow_fraction=kwargs.get("min_shadow_fraction", 0.015),
            min_oil_fraction=kwargs.get("min_oil_fraction", 0.04),
            include_unshadowed=kwargs.get("include_unshadowed", False),
        )
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

        shadowed_tanks = sum(estimate.has_internal_shadow for estimate in estimates)
        oil_tanks = sum(estimate.has_oil_evidence for estimate in estimates)
        measured_tanks = sum(estimate.included_in_volume for estimate in estimates)
        print(
            f"Open-roof evidence found in {measured_tanks}/{len(estimates)} tanks "
            f"({shadowed_tanks} with shadow, {oil_tanks} with oil evidence). "
            f"Volume calculated for {measured_tanks}/{len(estimates)} tanks using "
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


if __name__ == "__main__":
    DATASET_ROOT = "/Users/alfeu/.cache/kagglehub/datasets/towardsentropy/oil-storage-tanks/versions/1/Oil Tanks"
    DEFAULT_IMAGE = os.path.join(DATASET_ROOT, "large_images/01_large.jpg")

    parser = argparse.ArgumentParser(
        description="Sliced Classic Vision Oil Tank Detector"
    )
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE, help="Path to the input image"
    )
    parser.add_argument("--output", default="predictions/classic_vision_sliced.png")
    parser.add_argument(
        "--method",
        choices=["hough", "contours", "combined"],
        default="combined",
        help="Detection method (default: combined)",
    )
    parser.add_argument("--slice-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.1)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--block-size", type=int, default=11)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-area", type=int, default=10000)
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--max-radius", type=int, default=100)
    parser.add_argument(
        "--calculate-volume",
        action="store_true",
        help="Estimate tank volume from internal shadow area",
    )
    parser.add_argument(
        "--volume-csv",
        default=None,
        help="Path for per-tank volume CSV (default: output filename + _volumes.csv)",
    )
    parser.add_argument(
        "--shadow-mask-output",
        default=None,
        help="Path for the open-roof evidence mask image",
    )
    parser.add_argument(
        "--min-shadow-fraction",
        type=float,
        default=0.015,
        help="Minimum internal shadow fraction required to mark a tank as shadowed",
    )
    parser.add_argument(
        "--min-oil-fraction",
        type=float,
        default=0.04,
        help="Minimum dark oil/liquid fraction required to mark a tank as open roof",
    )
    parser.add_argument(
        "--include-unshadowed",
        action="store_true",
        help="Also calculate 100% volume for tanks without measurable internal shadow",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save intermediate preprocessing images"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
    else:
        process_image_sliced(
            args.image,
            args.output,
            method=args.method,
            slice_size=args.slice_size,
            overlap=args.overlap,
            invert=args.invert,
            block_size=args.block_size,
            min_area=args.min_area,
            max_area=args.max_area,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            calculate_volume=args.calculate_volume,
            volume_csv_path=args.volume_csv,
            evidence_mask_path=args.shadow_mask_output,
            min_shadow_fraction=args.min_shadow_fraction,
            min_oil_fraction=args.min_oil_fraction,
            include_unshadowed=args.include_unshadowed,
            debug=args.debug,
        )
