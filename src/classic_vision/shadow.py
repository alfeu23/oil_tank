import csv
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class TankVolumeEstimate:
    tank_id: int
    x: int
    y: int
    radius: int
    score: float
    external_area: int
    internal_shadow_area: int
    internal_shadow_ratio: float
    oil_area: int
    oil_ratio: float
    volume_ratio: float | None
    volume_percent: float | None
    has_internal_shadow: bool
    has_oil_evidence: bool
    roof_type: str
    included_in_volume: bool


def _circle_mask(shape, x, y, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    return mask


def _remove_small_components(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def _largest_component_shape(mask, reference_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = {
        "area_fraction": 0.0,
        "circularity": 0.0,
        "aspect_ratio": 0.0,
        "fill_ratio": 0.0,
    }

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component = np.where(labels == label_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circularity = 0.0
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * np.pi * cv2.contourArea(contours[0]) / (perimeter**2)

        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
        area_fraction = area / reference_area if reference_area > 0 else 0.0

        if area_fraction > best["area_fraction"]:
            best = {
                "area_fraction": float(area_fraction),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "fill_ratio": float(fill_ratio),
            }

    return best


def _component_stats(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component = labels == label_id
        components.append(
            {
                "area": area,
                "bbox": (y, x, y + h, x + w),
                "image": component[y : y + h, x : x + w],
                "label_id": label_id,
            }
        )
    return labels, components


def _bbox_intersection_area(bb1, bb2):
    y_min1, x_min1, y_max1, x_max1 = bb1
    y_min2, x_min2, y_max2, x_max2 = bb2
    x_left = max(x_min1, x_min2)
    x_right = min(x_max1, x_max2)
    y_top = max(y_min1, y_min2)
    y_bottom = min(y_max1, y_max2)
    return max(0, x_right - x_left) * max(0, y_bottom - y_top)


def _clear_border(mask):
    labels, components = _component_stats(mask)
    cleaned = mask.copy()
    h, w = mask.shape[:2]
    for component in components:
        y_min, x_min, y_max, x_max = component["bbox"]
        if y_min <= 0 or x_min <= 0 or y_max >= h or x_max >= w:
            cleaned[labels == component["label_id"]] = 0
    return cleaned


def _fill_small_holes(mask, max_hole_area=64):
    inverse = np.where(mask > 0, 0, 255).astype(np.uint8)
    labels, holes = _component_stats(inverse)
    filled = mask.copy()
    h, w = mask.shape[:2]
    for hole in holes:
        y_min, x_min, y_max, x_max = hole["bbox"]
        touches_border = y_min <= 0 or x_min <= 0 or y_max >= h or x_max >= w
        if not touches_border and hole["area"] <= max_hole_area:
            filled[labels == hole["label_id"]] = 255
    return filled


def _threshold_minimum_approx(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    hist, bin_edges = np.histogram(values, bins=256)
    hist = hist.astype(np.float32)

    for _ in range(128):
        maxima = []
        for i in range(1, len(hist) - 1):
            if hist[i - 1] < hist[i] and hist[i + 1] < hist[i]:
                maxima.append(i)
        if len(maxima) <= 2:
            break
        hist = np.convolve(hist, np.array([1, 1, 1], dtype=np.float32) / 3.0, "same")

    if len(maxima) >= 2:
        left, right = sorted(maxima, key=lambda idx: hist[idx], reverse=True)[:2]
        left, right = sorted((left, right))
        valley = left + int(np.argmin(hist[left : right + 1]))
        return float((bin_edges[valley] + bin_edges[valley + 1]) / 2.0)

    normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(values.min() + (otsu / 255.0) * (values.max() - values.min()))


def _kaggle_shadow_enhancement(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0] * (100.0 / 255.0)
    blue_yellow = lab[:, :, 2] - 128.0
    return -(lightness + blue_yellow) / (value + 1.0)


def detect_kaggle_shadow_regions(
    image,
    circle,
    factor_x=0.5,
    factor_y=0.6,
    min_region_area=25,
):
    x, y, r = [int(v) for v in circle[:3]]
    fallback_area = int(np.count_nonzero(_circle_mask(image.shape, x, y, r)))
    x_min = max(0, x - r)
    x_max = min(image.shape[1], x + r)
    y_min = max(0, y - r)
    y_max = min(image.shape[0], y + r)
    if x_max <= x_min or y_max <= y_min:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0, fallback_area, 0.0, False

    margin_x = int((x_max - x_min) * factor_x)
    margin_y = int((y_max - y_min) * factor_y)
    crop_x_min = max(0, x_min - margin_x)
    crop_x_max = min(image.shape[1], x_max + margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_y_max = min(image.shape[0], y_max + margin_y // 2)
    crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    if crop.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0, fallback_area, 0.0, False

    bbox_relative = (
        y_min - crop_y_min,
        x_min - crop_x_min,
        y_max - crop_y_min,
        x_max - crop_x_min,
    )

    enhanced = _kaggle_shadow_enhancement(crop)
    threshold_min = _threshold_minimum_approx(enhanced)
    threshold_mean = float(np.mean(enhanced))
    threshold = (0.6 * threshold_min) + (0.4 * threshold_mean)
    thresh_mask = np.where(enhanced > threshold, 255, 0).astype(np.uint8)

    thresh_mask = _clear_border(thresh_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    processed = _fill_small_holes(processed, max_hole_area=64)

    labels, components = _component_stats(processed)
    regions = []
    bbox_area = max(1, (bbox_relative[2] - bbox_relative[0]) * (bbox_relative[3] - bbox_relative[1]))
    min_intersection = max(120, int(bbox_area * 0.08))
    for component in components:
        if component["area"] <= min_region_area:
            continue
        if _bbox_intersection_area(bbox_relative, component["bbox"]) <= min_intersection:
            continue
        y0, x0, y1, x1 = component["bbox"]
        threshold_mean_in_bbox = float(np.mean(thresh_mask[y0:y1, x0:x1] > 0))
        component_mean = float(np.mean(component["image"]))
        if abs(threshold_mean_in_bbox - component_mean) >= 0.12:
            continue
        regions.append(component)

    if not regions:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0, fallback_area, 0.0, False

    regions = sorted(regions, key=lambda region: region["area"], reverse=True)
    selected = regions[:2]
    external_area = int(selected[0]["area"])
    internal_area = int(selected[1]["area"]) if len(selected) > 1 else 0
    shadow_ratio = internal_area / external_area if external_area > 0 else 0.0

    crop_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for region in selected:
        crop_mask[labels == region["label_id"]] = 255

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = crop_mask
    return full_mask, internal_area, external_area, shadow_ratio, True


def detect_internal_shadow(
    image,
    circle,
    shadow_percentile=28,
    min_shadow_fraction=0.015,
    analysis_radius_scale=0.92,
):
    """
    Segment the dark internal floating-roof shadow for one detected tank.

    The mask is clipped to the tank circle, then thresholded locally so each
    tank is judged against its own roof brightness. The returned ratio is:

        internal shadow pixels / external tank pixels
    """
    x, y, r = [int(v) for v in circle[:3]]
    external_mask = _circle_mask(image.shape, x, y, r)
    analysis_radius = max(2, int(r * analysis_radius_scale))
    analysis_mask = _circle_mask(image.shape, x, y, analysis_radius)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    tank_pixels = analysis_mask > 0
    external_area = int(np.count_nonzero(external_mask))
    if external_area == 0 or not np.any(tank_pixels):
        return np.zeros(image.shape[:2], dtype=np.uint8), 0, external_area, 0.0

    local_l = l_channel[tank_pixels]
    local_v = value[tank_pixels]
    l_cutoff = float(np.percentile(local_l, shadow_percentile))
    v_cutoff = float(np.percentile(local_v, shadow_percentile))
    median_l = float(np.median(local_l))
    median_v = float(np.median(local_v))

    dark_l = l_channel <= min(l_cutoff, median_l - 8)
    dark_v = value <= min(v_cutoff, median_v - 6)
    shadow_mask = np.where((analysis_mask > 0) & dark_l & dark_v, 255, 0).astype(
        np.uint8
    )

    kernel_size = max(3, int(round(r * 0.10)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_component_area = max(4, int(external_area * min_shadow_fraction * 0.35))
    shadow_mask = _remove_small_components(shadow_mask, min_component_area)

    internal_shadow_area = int(np.count_nonzero(shadow_mask))
    internal_shadow_ratio = internal_shadow_area / external_area
    if internal_shadow_ratio < min_shadow_fraction:
        shadow_mask[:] = 0
        internal_shadow_area = 0
        internal_shadow_ratio = 0.0

    return shadow_mask, internal_shadow_area, external_area, internal_shadow_ratio


def detect_oil_evidence(
    image,
    circle,
    min_oil_fraction=0.30,
    analysis_radius_scale=0.78,
    max_white_roof_fraction=0.55,
    max_soil_fraction=0.30,
    max_green_fraction=0.20,
    max_internal_edge_density=0.20,
    min_oil_component_fraction=0.20,
    min_oil_component_circularity=0.60,
    min_oil_component_aspect=0.55,
    min_oil_component_fill=0.45,
):
    """
    Detect visible dark/liquid-like interior pixels inside a tank.

    This is roof-type evidence, not the volume measurement itself. A tank is
    treated as open roof only when the dark region is compact, sufficiently
    large, not mostly white roof, and not dominated by soil/vegetation color.
    """
    x, y, r = [int(v) for v in circle[:3]]
    external_mask = _circle_mask(image.shape, x, y, r)
    analysis_radius = max(2, int(r * analysis_radius_scale))
    analysis_mask = _circle_mask(image.shape, x, y, analysis_radius)
    external_area = int(np.count_nonzero(external_mask))
    if external_area == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0, 0.0, False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    tank_pixels = analysis_mask > 0
    if not np.any(tank_pixels):
        return np.zeros(image.shape[:2], dtype=np.uint8), 0, 0.0, False

    local_l = l_channel[tank_pixels]
    local_v = value[tank_pixels]
    local_s = saturation[tank_pixels]
    l_std = float(np.std(local_l))
    median_l = float(np.median(local_l))
    median_v = float(np.median(local_v))
    white_roof_ratio = float(np.mean((local_v > 190) & (local_s < 45)))

    green_pixels = (hue >= 35) & (hue <= 90) & (saturation >= 35)
    soil_pixels = (hue >= 5) & (hue <= 35) & (saturation >= 35) & (value < 220)
    local_green_ratio = float(np.mean(green_pixels[tank_pixels]))
    local_soil_ratio = float(np.mean(soil_pixels[tank_pixels]))

    absolute_oil = (
        (value <= 165)
        & (l_channel <= 180)
        & (saturation <= 120)
        & ~green_pixels
        & ~soil_pixels
    )
    relative_oil = (
        (l_std >= 12)
        & (l_channel <= median_l - 16)
        & (value <= median_v - 12)
        & ~green_pixels
        & ~soil_pixels
    )
    oil_mask = np.where((analysis_mask > 0) & (absolute_oil | relative_oil), 255, 0)
    oil_mask = oil_mask.astype(np.uint8)

    kernel_size = max(3, int(round(r * 0.08)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_component_area = max(4, int(external_area * min_oil_fraction * 0.30))
    oil_mask = _remove_small_components(oil_mask, min_component_area)

    oil_area = int(np.count_nonzero(oil_mask))
    oil_ratio = oil_area / external_area
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 50, 120)
    internal_edge_density = float(np.mean(edges[tank_pixels] > 0))
    component_shape = _largest_component_shape(oil_mask, external_area)

    has_oil_evidence = (
        oil_ratio >= min_oil_fraction
        and white_roof_ratio <= max_white_roof_fraction
        and local_soil_ratio <= max_soil_fraction
        and local_green_ratio <= max_green_fraction
        and internal_edge_density <= max_internal_edge_density
        and component_shape["area_fraction"] >= min_oil_component_fraction
        and component_shape["circularity"] >= min_oil_component_circularity
        and component_shape["aspect_ratio"] >= min_oil_component_aspect
        and component_shape["fill_ratio"] >= min_oil_component_fill
    )

    if not has_oil_evidence:
        oil_mask[:] = 0
        oil_area = 0
        oil_ratio = 0.0

    return oil_mask, oil_area, oil_ratio, has_oil_evidence


def estimate_tank_volumes(
    image,
    circles,
    scores=None,
    min_shadow_fraction=0.015,
    min_oil_fraction=0.30,
    include_unshadowed=False,
    classification_circles=None,
    min_open_roof_radius=24,
    max_white_roof_fraction=0.55,
    max_soil_fraction=0.30,
    max_green_fraction=0.20,
    max_internal_edge_density=0.20,
    min_oil_component_fraction=0.20,
    min_oil_component_circularity=0.60,
    min_oil_component_aspect=0.55,
    min_oil_component_fill=0.45,
    volume_method="kaggle_shadow",
    force_open_roof=False,
):
    estimates = []
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if classification_circles is None:
        classification_circles = circles

    for tank_id, circle in enumerate(circles, start=1):
        classification_circle = classification_circles[tank_id - 1]
        score = float(scores[tank_id - 1]) if scores is not None else 1.0
        if volume_method == "kaggle_shadow":
            (
                shadow_mask,
                shadow_area,
                external_area,
                shadow_ratio,
                has_shadow_regions,
            ) = detect_kaggle_shadow_regions(image, classification_circle)
        else:
            (
                shadow_mask,
                shadow_area,
                external_area,
                shadow_ratio,
            ) = detect_internal_shadow(
                image,
                circle,
                min_shadow_fraction=min_shadow_fraction,
            )
            has_shadow_regions = shadow_area > 0

        oil_mask, oil_area, oil_ratio, has_oil_evidence = detect_oil_evidence(
            image,
            classification_circle,
            min_oil_fraction=min_oil_fraction,
            max_white_roof_fraction=max_white_roof_fraction,
            max_soil_fraction=max_soil_fraction,
            max_green_fraction=max_green_fraction,
            max_internal_edge_density=max_internal_edge_density,
            min_oil_component_fraction=min_oil_component_fraction,
            min_oil_component_circularity=min_oil_component_circularity,
            min_oil_component_aspect=min_oil_component_aspect,
            min_oil_component_fill=min_oil_component_fill,
        )

        has_internal_shadow = shadow_area > 0
        cx, cy, cr = [int(v) for v in classification_circle[:3]]
        is_clipped = (
            cx - cr < 2
            or cy - cr < 2
            or cx + cr >= image.shape[1] - 2
            or cy + cr >= image.shape[0] - 2
        )
        if cr < min_open_roof_radius or is_clipped:
            oil_mask[:] = 0
            oil_area = 0
            oil_ratio = 0.0
            has_oil_evidence = False

        has_open_roof_evidence = (
            (force_open_roof or has_oil_evidence)
            and cr >= min_open_roof_radius
            and not is_clipped
        )
        if has_open_roof_evidence:
            combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, oil_mask)

        included_in_volume = include_unshadowed or has_open_roof_evidence
        roof_type = "open_roof" if has_open_roof_evidence else "closed_roof"
        if included_in_volume:
            volume_ratio = float(np.clip(1.0 - shadow_ratio, 0.0, 1.0))
            volume_percent = volume_ratio * 100.0
        else:
            volume_ratio = None
            volume_percent = None

        x, y, r = [int(v) for v in circle[:3]]
        estimates.append(
            TankVolumeEstimate(
                tank_id=tank_id,
                x=x,
                y=y,
                radius=r,
                score=score,
                external_area=int(external_area),
                internal_shadow_area=int(shadow_area),
                internal_shadow_ratio=float(shadow_ratio),
                oil_area=int(oil_area),
                oil_ratio=float(oil_ratio),
                volume_ratio=volume_ratio,
                volume_percent=volume_percent,
                has_internal_shadow=has_internal_shadow,
                has_oil_evidence=has_oil_evidence,
                roof_type=roof_type,
                included_in_volume=included_in_volume,
            )
        )

    return estimates, combined_mask


def draw_volume_estimates(image, estimates, shadow_mask=None):
    output = image.copy()
    if shadow_mask is not None and np.any(shadow_mask):
        shadow_overlay = np.zeros_like(output)
        shadow_overlay[:, :, 0] = shadow_mask
        output = cv2.addWeighted(output, 1.0, shadow_overlay, 0.55, 0)

    for estimate in estimates:
        color = (255, 120, 0) if estimate.included_in_volume else (160, 160, 160)
        cv2.circle(output, (estimate.x, estimate.y), estimate.radius, color, 2)
        cv2.circle(output, (estimate.x, estimate.y), 2, (0, 0, 255), 3)

        if estimate.included_in_volume:
            label = f"{estimate.tank_id}: {estimate.volume_percent:.0f}%"
        else:
            label = f"{estimate.tank_id}: closed"
        label_origin = (
            max(0, estimate.x - estimate.radius),
            max(14, estimate.y - estimate.radius - 6),
        )
        cv2.putText(
            output,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def write_volume_csv(path, estimates):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tank_id",
                "x",
                "y",
                "radius",
                "score",
                "external_area",
                "internal_shadow_area",
                "internal_shadow_ratio",
                "oil_area",
                "oil_ratio",
                "volume_ratio",
                "volume_percent",
                "has_internal_shadow",
                "has_oil_evidence",
                "roof_type",
                "included_in_volume",
            ],
        )
        writer.writeheader()
        for estimate in estimates:
            row = asdict(estimate)
            row["score"] = f"{estimate.score:.4f}"
            row["internal_shadow_ratio"] = f"{estimate.internal_shadow_ratio:.6f}"
            row["oil_ratio"] = f"{estimate.oil_ratio:.6f}"
            row["volume_ratio"] = (
                f"{estimate.volume_ratio:.6f}"
                if estimate.volume_ratio is not None
                else ""
            )
            row["volume_percent"] = (
                f"{estimate.volume_percent:.2f}"
                if estimate.volume_percent is not None
                else ""
            )
            writer.writerow(row)
