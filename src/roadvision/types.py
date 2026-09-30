from dataclasses import dataclass
from typing import Optional


@dataclass
class DamageDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    road_overlap: float = 0.0
    relative_depth: Optional[float] = None
    depth_contrast: Optional[float] = None
    severity_score: Optional[float] = None
    severity_label: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
