from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .gps import GPSTrack
from .models.depth import RelativeDepthEstimator, defect_depth_features
from .models.detector import RoadDamageDetector
from .models.segmenter import RoadSegmenter
from .severity import SeverityWeights, score_detection
from .types import DamageDetection
from .visualize import draw_detection, overlay_road_mask


class RoadVisionPipeline:
    def __init__(
        self,
        detector: RoadDamageDetector,
        segmenter: RoadSegmenter,
        depth_estimator: RelativeDepthEstimator,
        severity_weights: SeverityWeights,
        min_road_overlap: float = 0.35,
        segmentation_every_n: int = 3,
        depth_every_n: int = 3,
        gps_track: Optional[GPSTrack] = None,
        draw_road_mask: bool = True,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.severity_weights = severity_weights
        self.min_road_overlap = min_road_overlap
        self.segmentation_every_n = max(1, segmentation_every_n)
        self.depth_every_n = max(1, depth_every_n)
        self.gps_track = gps_track
        self.draw_road_mask = draw_road_mask
        self._road_mask: Optional[np.ndarray] = None
        self._depth_map: Optional[np.ndarray] = None

    @staticmethod
    def _road_overlap(mask: np.ndarray, det: DamageDetection) -> float:
        h, w = mask.shape[:2]
        x1, x2 = int(np.clip(det.x1, 0, w - 1)), int(np.clip(det.x2, 1, w))
        y1, y2 = int(np.clip(det.y1, 0, h - 1)), int(np.clip(det.y2, 1, h))
        roi = mask[y1:y2, x1:x2]
        return float(roi.mean()) if roi.size else 0.0

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> tuple[np.ndarray, list[DamageDetection]]:
        if self._road_mask is None or frame_index % self.segmentation_every_n == 0:
            self._road_mask = self.segmenter.predict(frame)
        if self._depth_map is None or frame_index % self.depth_every_n == 0:
            self._depth_map = self.depth_estimator.predict(frame)

        detections = self.detector.predict(frame)
        road_area = int(self._road_mask.sum()) if self._road_mask is not None else frame.shape[0] * frame.shape[1]
        gps = self.gps_track.at(timestamp_s) if self.gps_track else None

        kept: list[DamageDetection] = []
        for det in detections:
            det.road_overlap = self._road_overlap(self._road_mask, det)
            if det.road_overlap < self.min_road_overlap:
                continue

            if self._depth_map is not None:
                det.relative_depth, det.depth_contrast = defect_depth_features(
                    self._depth_map, (det.x1, det.y1, det.x2, det.y2)
                )

            det.severity_score, det.severity_label = score_detection(
                det,
                road_area=road_area,
                frame_width=frame.shape[1],
                weights=self.severity_weights,
            )

            if gps is not None:
                det.latitude = gps.lat
                det.longitude = gps.lon
            kept.append(det)

        annotated = frame.copy()
        if self.draw_road_mask and self._road_mask is not None:
            annotated = overlay_road_mask(annotated, self._road_mask)
        for det in kept:
            draw_detection(annotated, det)
        return annotated, kept


def analyse_video(
    source: str,
    output_path: str,
    events_path: str,
    pipeline: RoadVisionPipeline,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(events_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    fieldnames = [
        "time_s", "track_id", "class_name", "confidence", "road_overlap",
        "relative_depth", "depth_contrast", "severity_score", "severity_label",
        "latitude", "longitude", "x1", "y1", "x2", "y2",
    ]

    seen_tracks: set[tuple[int | None, str]] = set()
    frame_index = 0
    try:
        with open(events_path, "w", newline="", encoding="utf-8") as handle:
            out = csv.DictWriter(handle, fieldnames=fieldnames)
            out.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_s = frame_index / fps
                annotated, detections = pipeline.process_frame(frame, frame_index, timestamp_s)
                writer.write(annotated)

                for det in detections:
                    key = (det.track_id, det.class_name)
                    if det.track_id is not None and key in seen_tracks:
                        continue
                    if det.track_id is not None:
                        seen_tracks.add(key)
                    out.writerow({
                        "time_s": round(timestamp_s, 3),
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 4),
                        "road_overlap": round(det.road_overlap, 4),
                        "relative_depth": None if det.relative_depth is None else round(det.relative_depth, 4),
                        "depth_contrast": None if det.depth_contrast is None else round(det.depth_contrast, 4),
                        "severity_score": det.severity_score,
                        "severity_label": det.severity_label,
                        "latitude": det.latitude,
                        "longitude": det.longitude,
                        "x1": det.x1,
                        "y1": det.y1,
                        "x2": det.x2,
                        "y2": det.y2,
                    })
                frame_index += 1
    finally:
        cap.release()
        writer.release()
