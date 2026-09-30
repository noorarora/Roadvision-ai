from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .types import DamageDetection


@dataclass
class SeverityWeights:
    confidence: float = 0.20
    area: float = 0.30
    depth: float = 0.35
    lane: float = 0.15
    medium_threshold: float = 35.0
    high_threshold: float = 65.0


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def lane_risk(center_x: float, frame_width: int) -> float:
    """Simple centre-lane proxy. 1.0 near image centre, 0 at edges."""
    if frame_width <= 0:
        return 0.0
    normalized = center_x / frame_width
    return _clamp01(1.0 - abs(normalized - 0.5) * 2.0)


def area_risk(box_area: int, road_area: int) -> float:
    if road_area <= 0:
        return 0.0
    # Saturates at 8% of visible road area.
    return _clamp01((box_area / road_area) / 0.08)


def score_detection(
    detection: DamageDetection,
    road_area: int,
    frame_width: int,
    weights: SeverityWeights,
) -> tuple[float, str]:
    confidence = _clamp01(detection.confidence)
    area = area_risk(detection.area, road_area)
    depth = _clamp01(detection.depth_contrast or 0.0)
    lane = lane_risk(detection.center[0], frame_width)

    total_weight = weights.confidence + weights.area + weights.depth + weights.lane
    if total_weight <= 0:
        total_weight = 1.0

    score = 100.0 * (
        weights.confidence * confidence
        + weights.area * area
        + weights.depth * depth
        + weights.lane * lane
    ) / total_weight

    if score >= weights.high_threshold:
        label = "high"
    elif score >= weights.medium_threshold:
        label = "medium"
    else:
        label = "low"
    return round(float(score), 2), label
