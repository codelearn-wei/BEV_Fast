# Realtime Perception Subproject

This folder is a standalone engineering module for realtime multi-camera perception based on FastBEV.

## Goals

- Realtime frame ingestion from external systems.
- Unified six-camera packet protocol.
- Single pipeline supporting both `pth` and TensorRT (`trt`) backends.
- Practical runtime visualization (6-view + BEV obstacle panel + runtime stats).
- JSON result output per frame for downstream modules.

## Folder Structure

- `app.py`: main realtime pipeline entry.
- `frame_sources.py`: realtime input sources.
- `inferencers.py`: unified inferencer abstraction (`pth` / `trt`).
- `visualizer.py`: runtime visualization.
- `schemas.py`: frame packet schema and validation.
- `sender.py`: helper sender for TCP mode.
- `examples/template_frame.json`: input packet template.

## Input Packet Protocol

Each frame is one JSON object. Required fields:

- `camera_images`: dict of 6 camera image paths.
- `camera_infos`: dict of 6 camera calibration blocks.
- optional `frame_id`, `timestamp`, `ego2global_rotation`, `ego2global_translation`, `image_color`, `metadata`.

Camera names must be exactly:

- `CAM_FRONT_LEFT`
- `CAM_FRONT`
- `CAM_FRONT_RIGHT`
- `CAM_BACK_LEFT`
- `CAM_BACK`
- `CAM_BACK_RIGHT`

You can start from `examples/template_frame.json`.

## Run With PTH (current default)

Run realtime pipeline (directory source):

```bash
python projects/realtime_perception/app.py \
  --backend pth \
  --config configs/fastbev/paper/fastbev-r50-cbgs.py \
  --checkpoint model/fastbev-r50-cbgs.pth \
  --device cuda:0 \
  --source directory \
  --frame-dir outputs/realtime_perception/incoming \
  --out-dir outputs/realtime_perception/run \
  --show --save-frames
```

Then continuously drop packet JSON files into:

- `outputs/realtime_perception/incoming`

The pipeline polls this folder and processes new packets in timestamp order.

## Run With TCP Source (for true streaming)

Start receiver:

```bash
python projects/realtime_perception/app.py \
  --backend pth \
  --config configs/fastbev/paper/fastbev-r50-cbgs.py \
  --checkpoint model/fastbev-r50-cbgs.pth \
  --source tcp \
  --tcp-host 0.0.0.0 \
  --tcp-port 17999 \
  --out-dir outputs/realtime_perception/tcp_run
```

Send packets (JSON per line over TCP):

```bash
python projects/realtime_perception/sender.py \
  --host 127.0.0.1 \
  --port 17999 \
  --packet-dir outputs/realtime_perception/incoming \
  --fps 10
```

## TensorRT Adaptation

The pipeline is already backend-agnostic. Switch to TensorRT by changing args:

```bash
python projects/realtime_perception/app.py \
  --backend trt \
  --config configs/fastbev/paper/fastbev-r50-cbgs.py \
  --checkpoint model/fastbev-r50-cbgs.pth \
  --engine outputs/deploy/fastbev_fp16.engine \
  --source tcp
```

Note:

- `pth` mode is currently the default path for ongoing feature development.
- `trt` mode reuses existing `tools/fastbev_trt_runtime.py` custom image inferencer.

## Outputs

Inside `--out-dir`:

- `json/`: one inference result JSON per frame.
- `frames/`: rendered images if `--save-frames` enabled.
- `realtime_perception.mp4`: runtime video unless `--no-save-video`.

Each output JSON includes:

- backend, frame id, inference latency, detection count.
- 3D box detections (`center_xyz`, `size_xyz`, `yaw`, `score`, class name).
- sample metadata for integration with later modules.

## Development Notes

- This module is designed for engineering integration first.
- Keep using `pth` for rapid iteration.
- For deployment handoff, keep packet protocol unchanged and switch backend to `trt`.

