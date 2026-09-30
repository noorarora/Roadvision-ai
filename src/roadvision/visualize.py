from __future__ import annotations

import cv2
import numpy as np

from .types import DamageDetection


def overlay_road_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    if mask is None:
        return frame
    overlay = frame.copy()
    overlay[mask] = (overlay[mask].astype(np.float32) * 0.35 + np.array([0, 255, 0]) * 0.65).astype(np.uint8)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_detection(frame: np.ndarray, det: DamageDetection) -> None:
    palette = {
        "low": (0, 200, 0),
        "medium": (0, 180, 255),
        "high": (0, 0, 255),
    }
    colour = palette.get(det.severity_label or "", (255, 180, 0))
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), colour, 2)
    track = f" #{det.track_id}" if det.track_id is not None else ""
    sev = f" | {det.severity_label} {det.severity_score:.0f}" if det.severity_score is not None else ""
    text = f"{det.class_name}{track} {det.confidence:.2f}{sev}"
    cv2.putText(frame, text, (det.x1, max(18, det.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
