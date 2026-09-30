# RoadVision AI

AI-powered road damage detection and condition monitoring built as a modern continuation of my earlier pothole-detection project.

## What it does

RoadVision AI combines multiple computer-vision components into one inspection pipeline:

1. **Road segmentation** — SegFormer isolates the drivable road surface.
2. **Road-damage detection** — a custom YOLO26 model detects potholes and, when trained, other defect classes such as cracks, patches and damaged road edges.
3. **Multi-object tracking** — ByteTrack assigns persistent IDs across video frames so the same defect is not counted repeatedly.
4. **Monocular depth estimation** — Depth Anything V2 estimates relative scene depth around each defect. MiDaS can be added as an optional legacy backend for comparison.
5. **Severity scoring** — combines detection confidence, road-relative size, depth contrast and lane/road position into a reproducible 0–100 risk score.
6. **GPS sidecar support** — attaches time-aligned latitude/longitude from a CSV rather than relying on IP geolocation.
7. **Inspection output** — annotated video plus CSV defect records that can feed a dashboard or map.

> Depth is treated as **relative depth** by default. The project does not claim centimetre-accurate pothole depth unless a metric-depth model or camera calibration is supplied.

## Why this is a stronger rebuild

The original version used YOLOv4-tiny and OpenCV DNN. This rebuild keeps the practical idea — live/video pothole detection and coordinates — while upgrading the stack to a modular 2026 pipeline with segmentation, tracking, depth and severity analysis.

## Architecture

```text
Video / camera
    |
    +--> SegFormer road mask -------------------------+
    |                                                |
    +--> YOLO26 custom road-damage detector --> ByteTrack
                                                     |
Depth Anything V2 -----------------------------------+
                                                     |
GPS sidecar -----------------------------------------+
                                                     v
                                      severity + de-duplication
                                                     |
                           annotated video + CSV inspection log
```

## Project structure

```text
roadvision-ai/
├── app.py
├── configs/default.yaml
├── src/roadvision/
│   ├── cli.py
│   ├── config.py
│   ├── gps.py
│   ├── pipeline.py
│   ├── severity.py
│   ├── types.py
│   ├── visualize.py
│   └── models/
│       ├── depth.py
│       ├── detector.py
│       └── segmenter.py
└── tests/
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For local development:

```bash
pip install -e .
```

## Custom YOLO26 weights

The detector expects a custom Ultralytics checkpoint trained on road-damage labels, for example:

```text
weights/roadvision-yolo26n.pt
```

Train with your own dataset:

```bash
yolo detect train model=yolo26n.pt data=data/road_damage.yaml epochs=100 imgsz=640
```

A later experiment can compare YOLO26 against RT-DETR using the same train/validation split.

## Run on video

```bash
roadvision-analyse \
  --source input.mp4 \
  --weights weights/roadvision-yolo26n.pt \
  --output runs/demo.mp4 \
  --events runs/events.csv
```

Optional GPS sidecar:

```bash
roadvision-analyse \
  --source input.mp4 \
  --weights weights/roadvision-yolo26n.pt \
  --gps gps.csv
```

`gps.csv` format:

```csv
time_s,lat,lon
0.0,-34.9285,138.6007
5.0,-34.9287,138.6009
10.0,-34.9290,138.6012
```

Coordinates are linearly interpolated by video timestamp.

## Severity score

The current score is deliberately transparent and configurable. It uses:

- detection confidence
- bounding-box area as a fraction of the road region
- relative depth contrast between the defect and nearby road
- horizontal road position as a simple lane-risk proxy

This is a **risk score**, not a civil-engineering measurement. Future versions can replace it with calibrated physical measurements.

## Model choices

- **Detection:** Ultralytics YOLO26 custom checkpoint
- **Tracking:** ByteTrack through Ultralytics tracking
- **Road segmentation:** `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
- **Depth:** `depth-anything/Depth-Anything-V2-Small-hf`
- **Optional comparison:** MiDaS 3.1, RT-DETR

## Roadmap

- [x] Modular inference pipeline
- [x] Road mask filtering
- [x] Tracking hooks
- [x] Relative-depth severity signal
- [x] GPS sidecar interpolation
- [ ] Train custom YOLO26 road-damage checkpoint
- [ ] Add crack / patch / edge-damage classes
- [ ] Add RT-DETR benchmark
- [ ] Add metric-depth calibration experiment
- [ ] Add map/dashboard with road-segment condition scores
- [ ] Add before/after repair comparison
- [ ] Export ONNX / CoreML / TensorRT benchmarks

## Notes on licenses

This repository's own code can be licensed separately, but the models and libraries it loads have their own licenses. Check model/library terms before commercial deployment.
