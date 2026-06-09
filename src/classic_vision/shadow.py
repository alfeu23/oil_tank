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
):
    estimates = []
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if classification_circles is None:
        classification_circles = circles

    for tank_id, circle in enumerate(circles, start=1):
        classification_circle = classification_circles[tank_id - 1]
        score = float(scores[tank_id - 1]) if scores is not None else 1.0
        shadow_mask, shadow_area, external_area, shadow_ratio = detect_internal_shadow(
            image,
            circle,
            min_shadow_fraction=min_shadow_fraction,
        )
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

        has_open_roof_evidence = has_oil_evidence
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
