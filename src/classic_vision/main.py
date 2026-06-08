import argparse
import os
import sys

import cv2

# Add current directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hough import detect_by_contours, detect_circles, draw_circles
from l_channel import extract_l_channel, threshold_l_channel


def process_image(image_path, output_path=None, method="contours", save_mask=True):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing {image_path} using method: {method}...")

    # 1. Preprocessing
    l_channel = extract_l_channel(image)
    binary_mask = threshold_l_channel(l_channel)

    if save_mask:
        mask_path = output_path.replace(".png", "_mask.png")
        cv2.imwrite(mask_path, binary_mask)
        print(f"Binary mask saved to {mask_path}")

    # 2. Detection
    if method == "hough":
        circles = detect_circles(
            l_channel, min_dist=50, param1=50, param2=35, min_radius=10, max_radius=100
        )
    else:  # Default to contours
        circles = detect_by_contours(
            binary_mask, min_area=300, max_area=100000, circularity_threshold=0.6
        )

    print(f"Detected {len(circles)} tanks.")

    # 3. Draw and save results
    result_image = draw_circles(image, circles)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")


if __name__ == "__main__":
    DATASET_ROOT = "/Users/alfeu/.cache/kagglehub/datasets/towardsentropy/oil-storage-tanks/versions/1/Oil Tanks"
    DEFAULT_IMAGE_LARGE = os.path.join(DATASET_ROOT, "large_images/01_large.jpg")
    DEFAULT_IMAGE_SMALL = os.path.join(DATASET_ROOT, "image_patches/01_5_2.jpg")

    parser = argparse.ArgumentParser(description="Classic Vision Oil Tank Detector")
    parser.add_argument(
        "--image", help="Path to the input image", default=DEFAULT_IMAGE_SMALL
    )
    parser.add_argument(
        "--output",
        help="Path to save the output image",
        default="predictions/classic_vision_result.png",
    )
    parser.add_argument(
        "--method",
        choices=["hough", "contours"],
        default="contours",
        help="Detection method",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
    else:
        process_image(args.image, args.output, method=args.method)
