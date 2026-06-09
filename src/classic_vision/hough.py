import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_circles_hough(
    l_channel, min_dist=20, param1=80, param2=None, min_radius=10, max_radius=50
):
    """
    Detects circles using the Circular Hough Transform on the L channel.
    Returns array of shape (N, 3): [x, y, radius].
    """
    h, w = l_channel.shape[:2]
    scale_cap = max(min_radius, int(min(h, w) * 0.12))
    max_radius = max(min_radius, min(max_radius, scale_cap))
    if param2 is None:
        param2 = max(42, min(60, int(38 + 0.20 * max_radius)))

    blurred = cv2.GaussianBlur(l_channel, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        return np.round(circles[0]).astype(np.int32)
    return np.empty((0, 3), dtype=np.int32)


def detect_by_contours(
    binary_image,
    min_area=100,
    max_area=100_000,
    circularity_threshold=0.35,
    min_radius=None,
    max_radius=None,
):
    """
    Detects circular objects via contour analysis.
    Uses convex hull as a fallback for noisy / partially occluded edges.
    Returns array of shape (N, 3): [x, y, radius].
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter**2)

        # Try convex hull if the raw contour is noisy
        if circularity < circularity_threshold:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            hull_perim = cv2.arcLength(hull, True)
            if hull_perim > 0:
                circularity = 4 * np.pi * hull_area / (hull_perim**2)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if min_radius is not None and radius < min_radius:
            continue
        if max_radius is not None and radius > max_radius:
            continue

        fill_ratio = area / (np.pi * radius**2) if radius > 0 else 0

        if (
            circularity >= circularity_threshold
            and aspect_ratio >= 0.45
            and fill_ratio >= 0.25
        ):
            circles.append([int(cx), int(cy), int(radius), circularity])

    if not circles:
        return np.empty((0, 3), dtype=np.int32)

    circles = np.array(circles)
    return circles[:, :3].astype(np.int32)


# ---------------------------------------------------------------------------
# NMS with correct circle IoU
# ---------------------------------------------------------------------------


def _circle_iou(c1, c2):
    """
    Approximate Intersection-over-Union for two circles.
    Uses the analytical formula for the area of intersection of two circles.
    Returns a value in [0, 1].
    """
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    # No overlap
    if d >= r1 + r2:
        return 0.0

    # One circle fully inside the other
    if d <= abs(r1 - r2):
        smaller_area = np.pi * min(r1, r2) ** 2
        larger_area = np.pi * max(r1, r2) ** 2
        return smaller_area / larger_area

    # Partial overlap — lens formula
    r1, r2 = float(r1), float(r2)
    part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    part3 = 0.5 * np.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    intersection = part1 + part2 - part3
    union = np.pi * r1**2 + np.pi * r2**2 - intersection
    return intersection / union if union > 0 else 0.0


def nms_circles(circles, scores=None, iou_threshold=0.3, return_scores=False):
    """
    Non-Maximum Suppression for circles using true circular IoU.
    Keeps the highest-scoring circle when two overlap significantly.
    Returns array of shape (N, 3).
    """
    if len(circles) == 0:
        empty_circles = np.empty((0, 3), dtype=np.int32)
        empty_scores = np.array([], dtype=np.float32)
        return (empty_circles, empty_scores) if return_scores else empty_circles

    circles = np.array(circles, dtype=np.int32)
    if scores is None:
        scores = circles[:, 2].astype(np.float32)
    scores = np.array(scores, dtype=np.float32)

    order = sorted(
        range(len(circles)), key=lambda i: (scores[i], circles[i, 2]), reverse=True
    )
    circles = circles[order]
    scores = scores[order]

    keep = []
    keep_scores = []
    suppressed = np.zeros(len(circles), dtype=bool)

    for i in range(len(circles)):
        if suppressed[i]:
            continue
        keep.append(circles[i])
        keep_scores.append(scores[i])
        for j in range(i + 1, len(circles)):
            if suppressed[j]:
                continue
            if _circle_iou(circles[i], circles[j]) > iou_threshold:
                suppressed[j] = True

    keep = (
        np.array(keep, dtype=np.int32)
        if keep
        else np.empty((0, 3), dtype=np.int32)
    )
    keep_scores = np.array(keep_scores, dtype=np.float32)
    return (keep, keep_scores) if return_scores else keep


def _candidate_edge_map(l_channel):
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 1.2)
    median = float(np.median(blurred))
    low = int(max(25, 0.55 * median))
    high = int(min(180, max(70, 1.20 * median)))
    return cv2.Canny(blurred, low, high)


def _score_circle_candidate(
    l_channel, body_mask, edge_map, circle, source_bonus=0.0, source_image=None
):
    x, y, r = [int(v) for v in circle[:3]]
    if r <= 0:
        return None

    inside = np.zeros_like(l_channel, dtype=np.uint8)
    ring = np.zeros_like(l_channel, dtype=np.uint8)
    annulus = np.zeros_like(l_channel, dtype=np.uint8)

    cv2.circle(inside, (x, y), max(1, int(r * 0.85)), 255, -1)
    cv2.circle(ring, (x, y), r, 255, max(2, int(r * 0.10)))
    cv2.circle(annulus, (x, y), int(r * 1.45), 255, -1)
    cv2.circle(annulus, (x, y), int(r * 1.02), 0, -1)

    ring_pixels = np.count_nonzero(ring)
    inside_pixels = np.count_nonzero(inside)
    if ring_pixels == 0 or inside_pixels < max(20, int(np.pi * r * r * 0.18)):
        return None

    edge_support = np.count_nonzero(cv2.bitwise_and(edge_map, edge_map, mask=ring))
    edge_support = edge_support / ring_pixels
    edge_points = np.column_stack(np.where((edge_map > 0) & (ring > 0)))
    if edge_points.size:
        angles = np.arctan2(edge_points[:, 0] - y, edge_points[:, 1] - x)
        angle_bins = ((angles + np.pi) / (2 * np.pi) * 24).astype(np.int32)
        angle_bins = np.clip(angle_bins, 0, 23)
        rim_coverage = len(np.unique(angle_bins)) / 24.0
    else:
        rim_coverage = 0.0

    inner_values = l_channel[inside > 0]
    annulus_values = l_channel[annulus > 0]
    if inner_values.size == 0:
        return None

    l75 = np.percentile(l_channel, 75)
    l30 = np.percentile(l_channel, 30)
    bright_roof = float(np.mean(inner_values >= l75))
    dark_roof = float(np.mean(inner_values <= l30))
    mask_support = (
        float(np.count_nonzero(body_mask[inside > 0])) / inside_pixels
        if body_mask is not None
        else 0.0
    )
    roof_support = max(bright_roof, dark_roof, mask_support)

    if annulus_values.size:
        local_contrast = abs(
            float(np.median(inner_values)) - float(np.median(annulus_values))
        )
        shadow_support = float(np.mean(annulus_values <= l30))
    else:
        local_contrast = 0.0
        shadow_support = 0.0

    median_saturation = 0.0
    median_value = 255.0
    green_support = 0.0
    if source_image is not None:
        hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        inside_pixels_mask = inside > 0
        green_pixels = (
            (hue >= 35)
            & (hue <= 90)
            & (saturation >= 35)
            & inside_pixels_mask
        )
        median_saturation = float(np.median(saturation[inside_pixels_mask]))
        median_value = float(np.median(value[inside_pixels_mask]))
        green_support = float(np.count_nonzero(green_pixels)) / inside_pixels

        if edge_support < 0.14 and local_contrast < 70:
            return None
        is_clipped = (
            x - r < 2
            or y - r < 2
            or x + r >= l_channel.shape[1] - 2
            or y + r >= l_channel.shape[0] - 2
        )
        min_rim_coverage = 0.35 if is_clipped else 0.52
        if rim_coverage < min_rim_coverage:
            return None
        if local_contrast < 30:
            return None
        if median_value < 145 and local_contrast < 65:
            return None
        if median_saturation > 58 and local_contrast < 95:
            return None
        if green_support > 0.22 and local_contrast < 110:
            return None

    score = source_bonus
    score += min(edge_support / 0.22, 1.0) * 2.3
    score += min(rim_coverage / 0.75, 1.0) * 0.7
    score += min(local_contrast / 35.0, 1.0) * 1.0
    score += min(roof_support / 0.45, 1.0) * 1.1
    score += min(shadow_support / 0.35, 1.0) * 0.8

    if source_image is not None:
        score += 0.45 if median_saturation <= 45 else -0.75
        score += 0.45 if median_value >= 155 else -0.65
        score += 0.35 if green_support <= 0.12 else -0.90

    if edge_support < 0.10 and source_bonus < 0.5:
        return None
    if roof_support < 0.16 and local_contrast < 12 and shadow_support < 0.18:
        return None
    min_score = 5.0 if source_image is not None else 2.0
    if score < min_score:
        return None

    return score


# ---------------------------------------------------------------------------
# Dual-method fusion
# ---------------------------------------------------------------------------


def detect_combined(
    l_channel,
    binary_mask,
    min_radius=10,
    max_radius=50,
    min_area=100,
    max_area=100_000,
    source_image=None,
):
    """
    Runs both Hough and contour detection, then fuses the results.

    A circle found by both methods gets a confidence boost (score = 2),
    one found by only one method gets score = 1. After NMS the highest-
    scoring detections are kept, helping reject single-method false positives.

    Returns:
        circles  – np.ndarray (N, 3) [x, y, radius]
        scores   – np.ndarray (N,)   confidence per circle (1 or 2)
    """
    hough = detect_circles_hough(
        l_channel,
        min_dist=max(10, min_radius * 2),
        param1=80,
        param2=None,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    contours = detect_by_contours(
        binary_mask,
        min_area=min_area,
        max_area=max_area,
        circularity_threshold=0.25,
        min_radius=min_radius,
        max_radius=max_radius,
    )

    edge_map = _candidate_edge_map(l_channel)
    contour_matches = np.zeros(len(contours), dtype=bool)
    candidates = []

    for c in hough:
        source_bonus = 0.2
        for i, d in enumerate(contours):
            if _circle_iou(c, d) > 0.25:
                source_bonus = 1.2
                contour_matches[i] = True
                break
        score = _score_circle_candidate(
            l_channel,
            binary_mask,
            edge_map,
            c,
            source_bonus=source_bonus,
            source_image=source_image,
        )
        if score is not None:
            candidates.append((c, score))

    for i, d in enumerate(contours):
        if contour_matches[i]:
            continue
        score = _score_circle_candidate(
            l_channel,
            binary_mask,
            edge_map,
            d,
            source_bonus=0.7,
            source_image=source_image,
        )
        if score is not None:
            candidates.append((d, score))

    if not candidates:
        return np.empty((0, 3), dtype=np.int32), np.array([], dtype=np.int32)

    circles = np.array([c for c, _ in candidates], dtype=np.int32)
    scores = np.array([s for _, s in candidates], dtype=np.float32)
    return nms_circles(circles, scores=scores, iou_threshold=0.3, return_scores=True)


# ---------------------------------------------------------------------------
# Legacy single-method entry point (keeps main.py's --method flag working)
# ---------------------------------------------------------------------------


def detect_circles(
    image, min_dist=20, param1=50, param2=30, min_radius=10, max_radius=50
):
    """Thin wrapper kept for backward compatibility with main.py."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return detect_circles_hough(
        gray,
        min_dist=min_dist,
        param1=param1,
        param2=param2,
        min_radius=min_radius,
        max_radius=max_radius,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_circles(image, circles, scores=None):
    """
    Draws detected circles on the image.
    Circles confirmed by both methods are drawn in green;
    single-method detections are drawn in yellow.
    """
    output = image.copy()
    if len(circles) == 0:
        return output

    for i, c in enumerate(circles):
        x, y, r = int(c[0]), int(c[1]), int(c[2])
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        color = (0, 255, 0) if score >= 3.0 else (0, 200, 200)
        cv2.circle(output, (x, y), r, color, 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

    return output
