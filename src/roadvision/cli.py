from __future__ import annotations

import argparse

from .config import load_config
from .gps import GPSTrack
from .models.depth import RelativeDepthEstimator
from .models.detector import RoadDamageDetector
from .models.segmenter import RoadSegmenter
from .pipeline import RoadVisionPipeline, analyse_video
from .severity import SeverityWeights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoadVision AI video analysis")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="Custom YOLO26 road-damage weights")
    parser.add_argument("--output", default="runs/annotated.mp4")
    parser.add_argument("--events", default="runs/events.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gps", default=None, help="Optional time_s,lat,lon CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    det_cfg = cfg["detector"]
    seg_cfg = cfg["segmenter"]
    depth_cfg = cfg["depth"]
    sev_cfg = cfg["severity"]
    out_cfg = cfg.get("output", {})

    detector = RoadDamageDetector(
        args.weights,
        confidence=float(det_cfg.get("confidence", 0.35)),
        iou=float(det_cfg.get("iou", 0.5)),
        tracker=str(det_cfg.get("tracker", "bytetrack.yaml")),
        allowed_classes=det_cfg.get("allowed_classes") or None,
    )
    segmenter = RoadSegmenter(
        model_name=str(seg_cfg["model"]),
        road_label=str(seg_cfg.get("road_label", "road")),
    )
    depth = RelativeDepthEstimator(str(depth_cfg["model"]))
    gps = GPSTrack(args.gps) if args.gps else None

    severity = SeverityWeights(
        confidence=float(sev_cfg.get("confidence_weight", 0.20)),
        area=float(sev_cfg.get("area_weight", 0.30)),
        depth=float(sev_cfg.get("depth_weight", 0.35)),
        lane=float(sev_cfg.get("lane_weight", 0.15)),
        medium_threshold=float(sev_cfg.get("medium_threshold", 35)),
        high_threshold=float(sev_cfg.get("high_threshold", 65)),
    )

    pipeline = RoadVisionPipeline(
        detector=detector,
        segmenter=segmenter,
        depth_estimator=depth,
        severity_weights=severity,
        min_road_overlap=float(seg_cfg.get("min_overlap", 0.35)),
        segmentation_every_n=int(seg_cfg.get("every_n_frames", 3)),
        depth_every_n=int(depth_cfg.get("every_n_frames", 3)),
        gps_track=gps,
        draw_road_mask=bool(out_cfg.get("draw_road_mask", True)),
    )
    analyse_video(args.source, args.output, args.events, pipeline)


if __name__ == "__main__":
    main()
