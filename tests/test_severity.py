from roadvision.severity import SeverityWeights, lane_risk, area_risk, score_detection
from roadvision.types import DamageDetection


def test_lane_risk_is_highest_near_center():
    assert lane_risk(50, 100) > lane_risk(5, 100)


def test_area_risk_saturates():
    assert area_risk(1000, 10000) == 1.0


def test_score_returns_valid_label():
    det = DamageDetection(40, 40, 60, 60, 0.9, 0, "pothole", depth_contrast=0.8)
    score, label = score_detection(det, road_area=5000, frame_width=100, weights=SeverityWeights())
    assert 0 <= score <= 100
    assert label in {"low", "medium", "high"}
