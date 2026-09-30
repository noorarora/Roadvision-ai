from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class RoadSegmenter:
    """Cityscapes SegFormer wrapper returning a boolean road mask."""

    def __init__(self, model_name: str, road_label: str = "road"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        wanted = road_label.strip().lower()
        matches = [idx for idx, name in id2label.items() if name.strip().lower() == wanted]
        if not matches:
            raise ValueError(f"Road label '{road_label}' not found in model labels: {id2label}")
        self.road_id = matches[0]

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=frame_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        labels = logits.argmax(dim=1)[0].detach().cpu().numpy()
        return labels == self.road_id
