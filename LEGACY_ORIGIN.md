# Legacy project origin

This project is a modern rebuild of the earlier public repository:

`noor02arora/pothole-detection-`

The earlier implementation used YOLOv4-tiny through OpenCV DNN, supported video/camera input, wrote detection screenshots, and attached coordinates using IP geolocation.

RoadVision AI preserves the original practical goal while replacing the older model stack and improving the data pipeline:

- YOLOv4-tiny -> custom YOLO26 detector
- repeated frame detections -> ByteTrack track IDs
- no road context -> SegFormer road mask
- no depth signal -> Depth Anything V2 relative depth
- IP geolocation -> timestamp-aligned GPS sidecar
- raw boxes -> transparent severity scoring + CSV inspection events

The goal is not to pretend the 2023 system used these newer models; this repository explicitly documents the rebuild and its lineage.
