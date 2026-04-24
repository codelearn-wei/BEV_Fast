from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .schemas import CAMERA_ORDER, FramePacket

CLASS_COLORS = [
    (255, 140, 0),
    (70, 170, 255),
    (0, 200, 255),
    (40, 120, 255),
    (180, 110, 255),
    (180, 180, 180),
    (0, 220, 120),
    (40, 255, 200),
    (255, 90, 180),
    (120, 210, 255),
]


def _load_camera_image(path: str, width: int, height: int, title: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(image, 'missing image', (18, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    cv2.rectangle(image, (0, 0), (width, 34), (18, 18, 18), -1)
    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (245, 245, 245), 2, cv2.LINE_AA)
    return image


def compose_six_camera_grid(frame: FramePacket, panel_width: int = 540) -> np.ndarray:
    cam_h = int(panel_width * 9 / 16)
    images = []
    for cam_name in CAMERA_ORDER:
        path = frame.camera_images.get(cam_name, '')
        images.append(_load_camera_image(path, panel_width, cam_h, cam_name))

    top = np.concatenate(images[:3], axis=1)
    bottom = np.concatenate(images[3:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def metric_to_canvas(x: float, y: float, bounds: Tuple[float, float, float, float],
                     bev_size: int) -> Tuple[int, int]:
    f_min, f_max, l_min, l_max = bounds
    px = int((y - l_min) / (l_max - l_min) * (bev_size - 1))
    py = int((f_max - x) / (f_max - f_min) * (bev_size - 1))
    return px, py


def box_corners_2d(center_x: float, center_y: float, length: float,
                   width: float, yaw: float) -> np.ndarray:
    half_l = max(length, 1e-3) * 0.5
    half_w = max(width, 1e-3) * 0.5
    corners = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = corners @ rot.T
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotated


def render_bev_panel(detections: List[Dict],
                     bounds: Tuple[float, float, float, float],
                     bev_size: int = 920) -> np.ndarray:
    panel = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    grad = np.linspace(42, 12, bev_size, dtype=np.uint8)
    panel[:, :, 1] = grad[:, None]
    panel[:, :, 2] = (grad * 0.75).astype(np.uint8)[:, None]

    f_min, f_max, l_min, l_max = bounds
    for tick in np.arange(math.ceil(f_min / 10.0) * 10.0, f_max + 0.1, 10.0):
        p1 = metric_to_canvas(float(tick), l_min, bounds, bev_size)
        p2 = metric_to_canvas(float(tick), l_max, bounds, bev_size)
        cv2.line(panel, p1, p2, (45, 55, 65), 1, cv2.LINE_AA)
    for tick in np.arange(math.ceil(l_min / 10.0) * 10.0, l_max + 0.1, 10.0):
        p1 = metric_to_canvas(f_min, float(tick), bounds, bev_size)
        p2 = metric_to_canvas(f_max, float(tick), bounds, bev_size)
        cv2.line(panel, p1, p2, (45, 55, 65), 1, cv2.LINE_AA)

    axis_x1 = metric_to_canvas(0.0, l_min, bounds, bev_size)
    axis_x2 = metric_to_canvas(0.0, l_max, bounds, bev_size)
    axis_y1 = metric_to_canvas(f_min, 0.0, bounds, bev_size)
    axis_y2 = metric_to_canvas(f_max, 0.0, bounds, bev_size)
    cv2.line(panel, axis_x1, axis_x2, (105, 120, 135), 2, cv2.LINE_AA)
    cv2.line(panel, axis_y1, axis_y2, (105, 120, 135), 2, cv2.LINE_AA)

    for idx, det in enumerate(detections):
        center = det.get('center_xyz', [0.0, 0.0, 0.0])
        size = det.get('size_xyz', [0.1, 0.1, 0.1])
        yaw = float(det.get('yaw', 0.0))
        label = det.get('label_name', 'obj')
        score = float(det.get('score', 0.0))

        corners = box_corners_2d(
            center_x=float(center[0]),
            center_y=float(center[1]),
            length=float(size[0]),
            width=float(size[1]),
            yaw=yaw)
        corner_pixels = np.array([
            metric_to_canvas(float(pt[0]), float(pt[1]), bounds, bev_size)
            for pt in corners
        ], dtype=np.int32)

        color = CLASS_COLORS[idx % len(CLASS_COLORS)]
        overlay = panel.copy()
        cv2.fillConvexPoly(overlay, corner_pixels, color)
        panel = cv2.addWeighted(overlay, 0.20, panel, 0.80, 0.0)
        cv2.polylines(panel, [corner_pixels], True, color, 2, cv2.LINE_AA)

        center_px = metric_to_canvas(float(center[0]), float(center[1]), bounds,
                                     bev_size)
        front_x = float(center[0]) + math.cos(yaw) * float(size[0]) * 0.6
        front_y = float(center[1]) + math.sin(yaw) * float(size[0]) * 0.6
        front_px = metric_to_canvas(front_x, front_y, bounds, bev_size)
        cv2.circle(panel, center_px, 2, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, center_px, front_px, color, 2, cv2.LINE_AA,
                        tipLength=0.25)

        text = f'{label} {score:.2f}'
        cv2.putText(panel, text, (center_px[0] + 6, max(20, center_px[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                    cv2.LINE_AA)

    ego = np.array([
        [-2.2, -1.0],
        [2.2, -1.0],
        [4.3, 0.0],
        [2.2, 1.0],
        [-2.2, 1.0],
    ], dtype=np.float32)
    ego_pixels = np.array([
        metric_to_canvas(float(pt[0]), float(pt[1]), bounds, bev_size)
        for pt in ego
    ], dtype=np.int32)
    cv2.polylines(panel, [ego_pixels], True, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (bev_size, 38), (16, 16, 16), -1)
    cv2.putText(panel, 'BEV Obstacles', (14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


class RuntimeVisualizer:

    def __init__(self,
                 out_dir: Path,
                 fps: int,
                 show: bool,
                 save_frames: bool,
                 save_video: bool,
                 bounds: Tuple[float, float, float, float]) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.out_dir / 'frames'
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.show = show
        self.save_frames = save_frames
        self.save_video = save_video
        self.bounds = bounds
        self.video_writer = None

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.show:
            cv2.destroyAllWindows()

    def render(self,
               frame: FramePacket,
               payload: Dict,
               avg_ms: float,
               frame_idx: int) -> np.ndarray:
        camera_grid = compose_six_camera_grid(frame)
        bev_panel = render_bev_panel(payload.get('detections', []), self.bounds)

        if bev_panel.shape[0] != camera_grid.shape[0]:
            bev_panel = cv2.resize(bev_panel, (bev_panel.shape[1], camera_grid.shape[0]))

        fused = np.concatenate([camera_grid, bev_panel], axis=1)
        footer = np.zeros((90, fused.shape[1], 3), dtype=np.uint8)
        footer[:] = (18, 18, 18)
        line1 = (
            f'frame={frame_idx} id={frame.frame_id} '
            f'infer={payload["inference_ms"]:.2f}ms avg={avg_ms:.2f}ms '
            f'fps={1000.0 / max(avg_ms, 1e-6):.2f}')
        line2 = (
            f'detections={payload["num_detections"]} '
            f'timestamp={frame.timestamp}')
        cv2.putText(footer, line1, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (242, 242, 242), 2, cv2.LINE_AA)
        cv2.putText(footer, line2, (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (215, 215, 215), 2, cv2.LINE_AA)
        output = np.concatenate([fused, footer], axis=0)

        if self.save_video:
            if self.video_writer is None:
                path = self.out_dir / 'realtime_perception.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    str(path), fourcc, self.fps,
                    (output.shape[1], output.shape[0]))
            self.video_writer.write(output)

        if self.save_frames:
            frame_name = f'{frame_idx:06d}_{frame.frame_id}.jpg'
            cv2.imwrite(str(self.frames_dir / frame_name), output)

        if self.show:
            cv2.imshow('realtime_perception', output)
            cv2.waitKey(1)

        return output
