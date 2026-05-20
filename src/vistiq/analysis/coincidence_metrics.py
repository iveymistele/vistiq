"""Coincidence / overlap metrics for :class:`~vistiq.analysis.coincidence.CoincidenceDetector`.

Metrics are registered in :data:`COINCIDENCE_METRICS` so adding a new metric is a single
function plus one registry entry (similar to :class:`~vistiq.segment.analysis.RegionAnalyzer`
property lists).

- Mask metrics (``outline`` mode): pixel-wise overlap on binary region masks; any dimensionality.
- 2D bbox metrics (``bounding_box`` mode, or bbox-derived in outline): axis-aligned boxes
  in ``(min_row, min_col, max_row, max_col)`` skimage/regionprops format.

  - ``giou`` is Geometry IoU (enclosing-box background ratio)
  - ``diou`` / ``ciou`` / ``eiou`` are standard 2D detection bbox metrics.
  - All bbox-only metrics are 2D only; 3D bounding boxes raise :class:`ValueError`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, FrozenSet, Literal, Tuple, Union

import numpy as np

# skimage regionprops 2D bbox: (min_row, min_col, max_row, max_col), max exclusive
BBox2D = Tuple[float, float, float, float]
BBox = Union[BBox2D, Tuple[float, ...]]

MetricKind = Literal["mask", "bbox_2d"]

_EPS = 1e-7 # epsilon constant for numerical stability (DIoU, CIoU, EIoU formulas where may divide by 0)


def _as_bbox_2d(bbox: BBox) -> BBox2D:
    """Normalize a regionprops bbox to 2D; raise for 3D boxes."""
    if len(bbox) == 4:
        return tuple(float(x) for x in bbox)  # type: ignore[return-value]
    if len(bbox) == 6:
        raise ValueError(
            f"Metric requires 2D axis-aligned bounding boxes (4 values), got 6 (3D). "
            "Use mask-based metrics (iou, dice, enclosure) in outline mode for 3D labels, "
            "or project regions to 2D before using giou/diou/ciou/eiou."
        )
    raise ValueError(
        f"Invalid bounding box length {len(bbox)}; expected 4 (2D) or 6 (3D)."
    )


def _bbox_areas_and_intersection(
    bbox1: BBox2D, bbox2: BBox2D
) -> Tuple[float, float, float, float]:
    """Return area1, area2, intersection area, and union area for two 2D boxes."""
    min_r1, min_c1, max_r1, max_c1 = bbox1
    min_r2, min_c2, max_r2, max_c2 = bbox2

    area1 = max(0.0, max_r1 - min_r1) * max(0.0, max_c1 - min_c1)
    area2 = max(0.0, max_r2 - min_r2) * max(0.0, max_c2 - min_c2)

    inter_r1 = max(min_r1, min_r2)
    inter_c1 = max(min_c1, min_c2)
    inter_r2 = min(max_r1, max_r2)
    inter_c2 = min(max_c1, max_c2)

    inter_w = max(0.0, inter_c2 - inter_c1)
    inter_h = max(0.0, inter_r2 - inter_r1)
    inter = inter_w * inter_h

    union = area1 + area2 - inter
    return area1, area2, inter, union


def _bbox_xyxy(bbox: BBox2D) -> Tuple[float, float, float, float]:
    """Convert (min_row, min_col, max_row, max_col) to (x1, y1, x2, y2)."""
    min_row, min_col, max_row, max_col = bbox
    return min_col, min_row, max_col, max_row


def _enclosing_box_xyxy(b1: BBox2D, b2: BBox2D) -> Tuple[float, float, float, float]:
    x1a, y1a, x2a, y2a = _bbox_xyxy(b1)
    x1b, y1b, x2b, y2b = _bbox_xyxy(b2)
    return (
        min(x1a, x1b),
        min(y1a, y1b),
        max(x2a, x2b),
        max(y2a, y2b),
    )


def _box_center_wh(bbox: BBox2D) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (x1 + w / 2.0, y1 + h / 2.0, w, h)


# --- Mask overlap metrics (segmentation / outline mode) ---


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Intersection over union on binary masks (Jaccard index)."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return 0.0
    return float(intersection / union)


def mask_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Dice coefficient: ``2 * intersection / (area_a + area_b)`` on masks."""
    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return 0.0
    return float(2 * intersection / sum_masks)


def mask_enclosure(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Proportion of the smaller object enclosed by overlap: ``intersection / min(area)``."""
    intersection = np.sum(mask1 & mask2)
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    min_area = min(float(area1), float(area2))
    if min_area == 0:
        return 0.0
    return float(intersection / min_area)


# --- 2D axis-aligned bounding-box metrics ---


def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    """IoU between two 2D axis-aligned bounding boxes."""
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    _, _, inter, union = _bbox_areas_and_intersection(b1, b2)
    if union == 0:
        return 0.0
    return float(inter / union)


def bbox_dice(bbox1: BBox, bbox2: BBox) -> float:
    """Dice coefficient from 2D box areas: ``2 * intersection / (area_a + area_b)``."""
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    area1, area2, inter, _ = _bbox_areas_and_intersection(b1, b2)
    denom = area1 + area2
    if denom == 0:
        return 0.0
    return float(2 * inter / denom)


def bbox_enclosure(bbox1: BBox, bbox2: BBox) -> float:
    """``intersection / min(area_a, area_b)`` for 2D boxes (1.0 when one box contains the other)."""
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    area1, area2, inter, _ = _bbox_areas_and_intersection(b1, b2)
    min_area = min(area1, area2)
    if min_area == 0:
        return 0.0
    return float(inter / min_area)


def bbox_giou(bbox1: BBox, bbox2: BBox) -> float:
    """Geometry IoU (GIoU): background fraction of the smallest enclosing box *C*.

    Let *C* be the axis-aligned box enclosing both regions, and *U* = area(A ∪ B).
    Geometry IoU is the fraction of *C* not occupied by either box:

        GIoU = (area(C) - area(A ∪ B)) / area(C)

    Returns 0.0 when the two boxes coincide (they fill *C*). Approaches 1.0 when both
    boxes are small and well separated inside a large *C*. This is **not** detection-style
    Generalized IoU (``IoU - (C - U) / C``).
    """
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    _, _, _, union = _bbox_areas_and_intersection(b1, b2)
    ex1, ey1, ex2, ey2 = _enclosing_box_xyxy(b1, b2)
    c_area = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
    if c_area <= 0:
        return 0.0
    exclusive = c_area - union
    return float(max(0.0, exclusive) / c_area)


def bbox_diou(bbox1: BBox, bbox2: BBox) -> float:
    """Distance IoU for 2D axis-aligned boxes."""
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    _, _, inter, union = _bbox_areas_and_intersection(b1, b2)
    if union == 0:
        return 0.0
    iou = inter / union
    cx1, cy1, _, _ = _box_center_wh(b1)
    cx2, cy2, _, _ = _box_center_wh(b2)
    ex1, ey1, ex2, ey2 = _enclosing_box_xyxy(b1, b2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if c2 <= 0:
        return float(iou)
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    return float(iou - rho2 / (c2 + _EPS))


def bbox_ciou(bbox1: BBox, bbox2: BBox) -> float:
    """Complete IoU for 2D axis-aligned boxes."""
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    _, _, inter, union = _bbox_areas_and_intersection(b1, b2)
    if union == 0:
        return 0.0
    iou = inter / union
    _, _, w1, h1 = _box_center_wh(b1)
    _, _, w2, h2 = _box_center_wh(b2)
    v = (4.0 / (math.pi**2)) * (math.atan(w2 / (h2 + _EPS)) - math.atan(w1 / (h1 + _EPS))) ** 2
    alpha = v / (1.0 - iou + v + _EPS)
    diou = bbox_diou(b1, b2)
    return float(diou - alpha * v)


def bbox_eiou(bbox1: BBox, bbox2: BBox) -> float:
    """Efficient IoU for 2D axis-aligned boxes."""
    b1, b2 = _as_bbox_2d(bbox1), _as_bbox_2d(bbox2)
    _, _, inter, union = _bbox_areas_and_intersection(b1, b2)
    if union == 0:
        return 0.0
    iou = inter / union
    cx1, cy1, w1, h1 = _box_center_wh(b1)
    cx2, cy2, w2, h2 = _box_center_wh(b2)
    ex1, ey1, ex2, ey2 = _enclosing_box_xyxy(b1, b2)
    cw = ex2 - ex1
    ch = ey2 - ey1
    c2 = cw**2 + ch**2 + _EPS
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    pw = (w1 - w2) ** 2 / (cw**2 + _EPS)
    ph = (h1 - h2) ** 2 / (ch**2 + _EPS)
    return float(iou - rho2 / c2 - pw - ph)


# Registry: metric name -> (kind, mask_fn, bbox_fn). None means unsupported for that representation.
MaskMetricFn = Callable[[np.ndarray, np.ndarray], float]
BboxMetricFn = Callable[[BBox, BBox], float]


class _MetricSpec:
    __slots__ = ("kind", "mask_fn", "bbox_fn")

    def __init__(
        self,
        kind: MetricKind,
        mask_fn: MaskMetricFn | None,
        bbox_fn: BboxMetricFn | None,
    ):
        self.kind = kind
        self.mask_fn = mask_fn
        self.bbox_fn = bbox_fn


# Central registry — add new metrics here.
_METRIC_SPECS: Dict[str, _MetricSpec] = {
    "iou": _MetricSpec("mask", mask_iou, bbox_iou),
    "dice": _MetricSpec("mask", mask_dice, bbox_dice),
    "enclosure": _MetricSpec("mask", mask_enclosure, bbox_enclosure),
    "giou": _MetricSpec("bbox_2d", None, bbox_giou),
    "diou": _MetricSpec("bbox_2d", None, bbox_diou),
    "ciou": _MetricSpec("bbox_2d", None, bbox_ciou),
    "eiou": _MetricSpec("bbox_2d", None, bbox_eiou),
}

COINCIDENCE_METRICS: Dict[str, MaskMetricFn | BboxMetricFn] = {
    "iou": mask_iou,
    "dice": mask_dice,
    "enclosure": mask_enclosure,
    "giou": bbox_giou,
    "diou": bbox_diou,
    "ciou": bbox_ciou,
    "eiou": bbox_eiou,
}

ALLOWED_COINCIDENCE_METRICS: FrozenSet[str] = frozenset(_METRIC_SPECS.keys())


def allowed_coincidence_metrics() -> list[str]:
    """Sorted list of registered coincidence metric names."""
    return sorted(ALLOWED_COINCIDENCE_METRICS)


def metric_kind(name: str) -> MetricKind:
    """Return whether a metric is mask-based or 2D-bbox-based."""
    return _METRIC_SPECS[name].kind


def compute_mask_metric(name: str, mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute a coincidence score from binary masks (outline mode)."""
    spec = _METRIC_SPECS[name]
    if spec.mask_fn is None:
        raise ValueError(
            f"Metric '{name}' is not available for mask/outline comparison. "
            f"Use bounding_box mode or a mask metric: "
            f"{sorted(k for k, s in _METRIC_SPECS.items() if s.mask_fn is not None)}"
        )
    return spec.mask_fn(mask1, mask2)


def compute_bbox_metric(name: str, bbox1: BBox, bbox2: BBox) -> float:
    """Compute a coincidence score from 2D (or 3D→rejected) axis-aligned bounding boxes."""
    spec = _METRIC_SPECS[name]
    if spec.bbox_fn is None:
        raise ValueError(f"Metric '{name}' has no bounding-box implementation.")
    return spec.bbox_fn(bbox1, bbox2)


def bbox_to_mask(bbox: BBox, shape: Tuple[int, ...]) -> np.ndarray:
    """Create a binary mask from a bounding box (same behavior as CoincidenceDetector._bbox_to_mask)."""
    if len(bbox) == 6:
        min_Z, min_Y, min_X, max_Z, max_Y, max_X = bbox
        if len(shape) == 3:
            mask = np.zeros(shape, dtype=bool)
            mask[min_Z:max_Z, min_Y:max_Y, min_X:max_X] = True
            return mask
        mask = np.zeros(shape, dtype=bool)
        mask[min_Y:max_Y, min_X:max_X] = True
        return mask
    min_row, min_col, max_row, max_col = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[min_row:max_row, min_col:max_col] = True
    return mask


def compute_bbox_metric_via_masks(
    name: str, bbox1: BBox, bbox2: BBox, shape: Tuple[int, ...]
) -> float:
    """Compute a mask-capable metric using box-filled masks (legacy bounding_box mode for iou/dice)."""
    if name not in ("iou", "dice"):
        return compute_bbox_metric(name, bbox1, bbox2)
    mask1 = bbox_to_mask(bbox1, shape)
    mask2 = bbox_to_mask(bbox2, shape)
    return compute_mask_metric(name, mask1, mask2)
