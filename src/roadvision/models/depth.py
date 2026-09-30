from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class RelativeDepthEstimator:
    """Depth Anything V2 wrapper producing normalized relative depth [0, 1]."""

    def __init__(self, model_name: str):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("depth-estimation", model=model_name, device=device)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = frame_bgr[:, :, ::-1]
        output = self.pipe(Image.fromarray(rgb))
        depth = np.asarray(output["depth"], dtype=np.float32)
        if depth.shape != frame_bgr.shape[:2]:
            import cv2
            depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        d_min = float(np.nanmin(depth))
        d_max = float(np.nanmax(depth))
        if d_max - d_min < 1e-8:
            return np.zeros_like(depth, dtype=np.float32)
        return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def defect_depth_features(depth: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return median defect depth and local road-relative depth contrast.

    The sign convention of monocular depth models can vary, so severity uses the
    absolute normalized contrast rather than claiming metric depth.
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 1, w))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 1, h))
    roi = depth[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0

    defect = float(np.median(roi))

    pad_x = max(8, (x2 - x1) // 2)
    pad_y = max(8, (y2 - y1) // 2)
    ax1, ax2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    ay1, ay2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
    local = depth[ay1:ay2, ax1:ax2]
    if local.size == 0:
        return defect, 0.0

    local_median = float(np.median(local))
    contrast = float(np.clip(abs(defect - local_median) / 0.25, 0.0, 1.0))
    return defect, contrast
