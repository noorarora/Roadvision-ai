from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from roadvision.types import DamageDetection


class RoadDamageDetector:
    """Custom YOLO26 road-damage detector with ByteTrack integration."""

    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.35,
        iou: float = 0.50,
        tracker: str = "bytetrack.yaml",
        allowed_classes: Iterable[int] | None = None,
    ):
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.iou = iou
        self.tracker = tracker
        self.allowed_classes = None if not allowed_classes else list(allowed_classes)

    def predict(self, frame: np.ndarray) -> list[DamageDetection]:
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            iou=self.iou,
            classes=self.allowed_classes,
            verbose=False,
        )[0]

        detections: list[DamageDetection] = []
        if result.boxes is None:
            return detections

        names = result.names
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for idx, coords in enumerate(xyxy):
            class_id = int(cls[idx])
            x1, y1, x2, y2 = [int(round(v)) for v in coords.tolist()]
            detections.append(
                DamageDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(conf[idx]),
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    track_id=int(ids[idx]) if ids is not None else None,
                )
            )
        return detections
