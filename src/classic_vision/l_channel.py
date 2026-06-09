import cv2
import numpy as np


def extract_l_channel(image):
    """
    Converts image to LAB color space and extracts the L channel.
    Applies CLAHE with a tile size scaled to image resolution.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    # Scale tile grid to image size so CLAHE is resolution-independent
    tile_h = max(4, image.shape[0] // 64)
    tile_w = max(4, image.shape[1] // 64)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))
    l_enhanced = clahe.apply(l_channel)

    return l_enhanced


def extract_saturation(image):
    """
    Extracts the saturation channel from HSV.
    Oil tanks (especially floating-roof tanks) often have a distinct
    saturation signature compared to surrounding ground.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    return s


def extract_edge_map(l_channel, low_thresh=None, high_thresh=None, dilate=False):
    """
    Computes a Canny edge map from the L channel.
    Tanks have strong, nearly circular edges (the rim).
    """
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 1.2)
    if low_thresh is None or high_thresh is None:
        median = float(np.median(blurred))
        low_thresh = int(max(25, 0.55 * median))
        high_thresh = int(min(180, max(70, 1.20 * median)))

    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def threshold_l_channel(l_channel, block_size=11, c=2, invert=False):
    """
    Applies Adaptive Thresholding and morphological cleanup.
    block_size must be odd and >= 3.
    """
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    block_size = max(3, block_size)

    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )

    if invert:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def build_combined_mask(image, block_size=11, invert=False, edge_weight=0.5):
    """
    Builds a compact tank-body mask for contour candidate generation.

    Bright oil-tank roofs are usually low-saturation circular regions. The
    previous adaptive-threshold/edge OR mask became almost entirely white on
    busy industrial patches, so contours had no useful objects to inspect.
    """
    l_channel = extract_l_channel(image)
    saturation = extract_saturation(image)

    light_cutoff = np.percentile(l_channel, 80)
    sat_cutoff = np.percentile(saturation, 85)
    roof_mask = np.where(
        (l_channel >= light_cutoff) & (saturation <= sat_cutoff), 255, 0
    ).astype(np.uint8)

    if invert:
        roof_mask = cv2.bitwise_not(roof_mask)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roof_mask = cv2.morphologyEx(
        roof_mask, cv2.MORPH_OPEN, cleanup_kernel, iterations=1
    )
    roof_mask = cv2.morphologyEx(
        roof_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=2
    )

    return roof_mask, l_channel


if __name__ == "__main__":
    pass
