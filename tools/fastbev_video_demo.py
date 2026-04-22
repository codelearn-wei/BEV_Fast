#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion

from mmcv import DictAction

from mmdet3d.datasets import build_dataset

from fastbev_infer import (get_sample_meta, normalize_device, prepare_cfg,
                           prepare_data, resolve_sample_index, run_inference)
from mmdet3d.apis import init_model

try:
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


CAMERA_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

CLASS_COLORS = [
    (255, 158, 0),
    (255, 99, 71),
    (255, 140, 0),
    (255, 215, 0),
    (0, 191, 255),
    (30, 144, 255),
    (0, 255, 127),
    (0, 206, 209),
    (255, 105, 180),
    (186, 85, 211),
]

PRIMARY_CLASSES = {
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
}

CLASS_COLOR_BY_NAME = {
    'car': (255, 140, 0),
    'truck': (70, 170, 255),
    'construction_vehicle': (0, 200, 255),
    'bus': (40, 120, 255),
    'trailer': (180, 110, 255),
    'barrier': (180, 180, 180),
    'motorcycle': (0, 220, 120),
    'bicycle': (40, 255, 200),
    'pedestrian': (255, 90, 180),
}

EGO_FORWARD_RANGE = 60.0
EGO_BACKWARD_RANGE = 15.0
EGO_SIDE_RANGE = 25.0


def alpha_blend(base, overlay, alpha):
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='FastBEV multi-frame visualization and video export')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='start sample index in cfg.data.test ann_file')
    parser.add_argument(
        '--start-token',
        type=str,
        default=None,
        help='start sample token in cfg.data.test ann_file')
    parser.add_argument(
        '--num-frames',
        type=int,
        default=30,
        help='number of sequential frames to visualize')
    parser.add_argument(
        '--stride', type=int, default=1, help='frame stride between samples')
    parser.add_argument(
        '--device', default='cuda:0', help='device for inference')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.35,
        help='score threshold for rendered boxes')
    parser.add_argument(
        '--topk',
        type=int,
        default=80,
        help='max number of detections to render')
    parser.add_argument(
        '--fps', type=int, default=6, help='fps for output video')
    parser.add_argument(
        '--bev-size',
        type=int,
        default=960,
        help='square canvas size for bev panels')
    parser.add_argument(
        '--cam-width',
        type=int,
        default=640,
        help='single camera panel width in output video')
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='number of warmup frames excluded from average latency')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs/fastbev_demo',
        help='directory for video and frame outputs')
    parser.add_argument(
        '--video-name',
        type=str,
        default='fastbev_demo.mp4',
        help='output video file name')
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='save rendered frames as images')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings with key=value pairs')
    return parser.parse_args()


def draw_title(image, title):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(panel, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (240, 240, 240), 2, cv2.LINE_AA)
    return panel


def load_and_resize_camera_image(path, camera_name, target_size):
    image = cv2.imread(str(path))
    if image is None:
        image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(image, 'image missing', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        image = cv2.resize(image, target_size)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (12, 12, 12), -1)
    cv2.putText(image, camera_name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def compose_camera_grid(sample_meta, cam_width):
    camera_files = sample_meta.get('camera_files', {})
    ordered_names = [name for name in CAMERA_ORDER if name in camera_files]
    if not ordered_names:
        ordered_names = sorted(camera_files.keys())

    cam_height = int(cam_width * 9 / 16)
    images = [
        load_and_resize_camera_image(camera_files[name], name,
                                     (cam_width, cam_height))
        for name in ordered_names
    ]
    while len(images) < 6:
        images.append(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))

    top_row = np.concatenate(images[:3], axis=1)
    bottom_row = np.concatenate(images[3:6], axis=1)
    camera_grid = np.concatenate([top_row, bottom_row], axis=0)
    return draw_title(camera_grid, 'Multi-camera Input')


def bev_feature_to_heatmap(bev_feature, bev_size):
    feature = bev_feature
    if isinstance(feature, (list, tuple)):
        feature = feature[0]
    if feature.ndim == 4:
        feature = feature[0]
    heat = feature.detach().float().abs().mean(dim=0).cpu().numpy()
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.percentile(heat, 99.0)
    if vmax <= 1e-6:
        vmax = heat.max() if heat.size > 0 else 1.0
    heat = np.clip(heat / max(vmax, 1e-6), 0.0, 1.0)
    heat = (heat * 255).astype(np.uint8)
    heat = cv2.resize(heat, (bev_size, bev_size), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return draw_title(heat, 'BEV Feature Heatmap')


def metric_to_canvas(points_xy, bounds, canvas_size):
    forward_min, forward_max, left_min, left_max = bounds
    width = float(max(left_max - left_min, 1e-6))
    height = float(max(forward_max - forward_min, 1e-6))
    px = (points_xy[:, 1] - left_min) / width * (canvas_size - 1)
    py = (forward_max - points_xy[:, 0]) / height * (canvas_size - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def draw_text_tag(image, text, anchor, color, font_scale=0.48):
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = int(anchor[0]), int(anchor[1])
    top_left = (x, max(0, y - text_h - baseline - 8))
    bottom_right = (min(image.shape[1] - 1, x + text_w + 10), y)
    cv2.rectangle(image, top_left, bottom_right, (12, 12, 12), -1)
    cv2.rectangle(image, top_left, bottom_right, color, 1)
    cv2.putText(image, text, (top_left[0] + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), 1,
                cv2.LINE_AA)
    return image


def draw_ego_vehicle(canvas, bounds, bev_size):
    ego = np.array([
        [-2.2, -1.0],
        [2.2, -1.0],
        [4.2, 0.0],
        [2.2, 1.0],
        [-2.2, 1.0],
    ], dtype=np.float32)
    ego_canvas = metric_to_canvas(ego, bounds, bev_size)
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, ego_canvas, (250, 250, 250))
    canvas[:] = alpha_blend(canvas, overlay, 0.22)
    cv2.polylines(canvas, [ego_canvas], True, (255, 255, 255), 3,
                  cv2.LINE_AA)
    center = metric_to_canvas(np.array([[0.0, 0.0]], dtype=np.float32),
                              bounds, bev_size)[0]
    forward = metric_to_canvas(np.array([[8.0, 0.0]], dtype=np.float32),
                               bounds, bev_size)[0]
    cv2.arrowedLine(canvas, tuple(center), tuple(forward), (255, 255, 255), 3,
                    cv2.LINE_AA, tipLength=0.2)


def render_bev_background(bounds, bev_size):
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    grad_y = np.linspace(42, 16, bev_size, dtype=np.uint8)
    canvas[:, :, 0] = (grad_y * 0.65).astype(np.uint8)[:, None]
    canvas[:, :, 1] = grad_y[:, None]
    canvas[:, :, 2] = (grad_y * 0.95).astype(np.uint8)[:, None]

    forward_min, forward_max, left_min, left_max = bounds
    max_radius = int(max(abs(forward_min), abs(forward_max), abs(left_min),
                         abs(left_max)))
    center = metric_to_canvas(np.array([[0.0, 0.0]], dtype=np.float32),
                              bounds, bev_size)[0]

    for meter in range(10, max_radius + 1, 10):
        circle_points = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False):
            circle_points.append(
                [meter * np.cos(angle), meter * np.sin(angle)])
        circle_points = metric_to_canvas(np.array(circle_points,
                                                  dtype=np.float32), bounds,
                                         bev_size)
        cv2.polylines(canvas, [circle_points], True, (60, 80, 88), 1,
                      cv2.LINE_AA)
        label_point = metric_to_canvas(
            np.array([[meter, 0.0]], dtype=np.float32), bounds, bev_size)[0]
        cv2.putText(canvas, f'{meter}m', tuple(label_point + np.array([6, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 190, 200), 1,
                    cv2.LINE_AA)

    forward_ticks = np.arange(
        np.ceil(forward_min / 10.0) * 10.0, forward_max + 0.1, 10.0)
    left_ticks = np.arange(
        np.ceil(left_min / 10.0) * 10.0, left_max + 0.1, 10.0)
    for value in left_ticks:
        line = metric_to_canvas(
            np.array([[forward_min, value], [forward_max, value]],
                     dtype=np.float32),
            bounds, bev_size)
        cv2.line(canvas, tuple(line[0]), tuple(line[1]), (44, 58, 64), 1,
                 cv2.LINE_AA)
    for value in forward_ticks:
        line = metric_to_canvas(
            np.array([[value, left_min], [value, left_max]], dtype=np.float32),
            bounds, bev_size)
        cv2.line(canvas, tuple(line[0]), tuple(line[1]), (44, 58, 64), 1,
                 cv2.LINE_AA)

    x_axis = metric_to_canvas(
        np.array([[forward_min, 0.0], [forward_max, 0.0]], dtype=np.float32), bounds,
        bev_size)
    y_axis = metric_to_canvas(
        np.array([[0.0, left_min], [0.0, left_max]], dtype=np.float32), bounds,
        bev_size)
    cv2.line(canvas, tuple(x_axis[0]), tuple(x_axis[1]), (110, 126, 138), 2,
             cv2.LINE_AA)
    cv2.line(canvas, tuple(y_axis[0]), tuple(y_axis[1]), (110, 126, 138), 2,
             cv2.LINE_AA)

    draw_ego_vehicle(canvas, bounds, bev_size)
    return canvas


def get_bev_bounds(_model):
    return (-EGO_BACKWARD_RANGE, EGO_FORWARD_RANGE, -EGO_SIDE_RANGE,
            EGO_SIDE_RANGE)


def estimate_ego_motion(dataset, sample_index):
    info = dataset.data_infos[sample_index]
    neighbors = []
    if sample_index > 0:
        prev_info = dataset.data_infos[sample_index - 1]
        if prev_info.get('scene_token') == info.get('scene_token'):
            neighbors.append(prev_info)
    if sample_index + 1 < len(dataset):
        next_info = dataset.data_infos[sample_index + 1]
        if next_info.get('scene_token') == info.get('scene_token'):
            neighbors.append(next_info)

    if not neighbors:
        return {
            'speed_mps': 0.0,
            'speed_kph': 0.0,
            'motion_xy': np.zeros(2, dtype=np.float32),
        }

    current_translation = np.array(
        info['ego2global_translation'][:2], dtype=np.float32)
    rotation = Quaternion(info['ego2global_rotation']).rotation_matrix[:2, :2]
    inv_rotation = rotation.T
    motions = []
    speeds = []

    for neighbor in neighbors:
        neighbor_translation = np.array(
            neighbor['ego2global_translation'][:2], dtype=np.float32)
        dt = abs(neighbor['timestamp'] - info['timestamp']) / 1e6
        if dt <= 1e-6:
            continue
        global_velocity = (neighbor_translation - current_translation) / dt
        ego_velocity = inv_rotation @ global_velocity
        motions.append(ego_velocity)
        speeds.append(float(np.linalg.norm(global_velocity)))

    if not motions:
        return {
            'speed_mps': 0.0,
            'speed_kph': 0.0,
            'motion_xy': np.zeros(2, dtype=np.float32),
        }

    motion_xy = np.mean(np.stack(motions, axis=0), axis=0)
    speed_mps = float(np.mean(speeds))
    return {
        'speed_mps': speed_mps,
        'speed_kph': speed_mps * 3.6,
        'motion_xy': motion_xy.astype(np.float32),
    }


def filter_detection_result(result, class_names, score_thr, topk):
    det = result['pts_bbox'] if 'pts_bbox' in result else result
    scores = det['scores_3d'].detach().cpu()
    labels = det['labels_3d'].detach().cpu()
    box_tensor = det['boxes_3d'].tensor.detach().cpu()
    corners = det['boxes_3d'].corners.detach().cpu()

    keep = scores >= score_thr
    class_keep = torch.tensor(
        [class_names[int(label)] in PRIMARY_CLASSES for label in labels],
        dtype=torch.bool)
    keep = keep & class_keep
    scores = scores[keep]
    labels = labels[keep]
    box_tensor = box_tensor[keep]
    corners = corners[keep]

    if topk is not None and scores.shape[0] > topk:
        order = torch_argsort_desc(scores)[:topk]
        scores = scores[order]
        labels = labels[order]
        box_tensor = box_tensor[order]
        corners = corners[order]
    return box_tensor.numpy(), corners.numpy(), scores.numpy(), labels.numpy()


def torch_argsort_desc(scores):
    return torch.argsort(scores, descending=True)


def render_bev_boxes_panel(result, class_names, bounds, bev_size, score_thr,
                           topk, ego_motion):
    canvas = render_bev_background(bounds, bev_size)
    forward_min, forward_max, left_min, left_max = bounds
    box_tensor, corners, scores, labels = filter_detection_result(
        result, class_names, score_thr, topk)
    bottom_indices = [0, 3, 7, 4]
    order = np.argsort(scores)
    legend_counts = {}

    for idx in order:
        box = box_tensor[idx]
        box_corners = corners[idx]
        score = scores[idx]
        label = labels[idx]
        label_name = class_names[int(label)]
        color = CLASS_COLOR_BY_NAME.get(
            label_name, CLASS_COLORS[int(label) % len(CLASS_COLORS)])
        bottom = box_corners[bottom_indices, :2]
        bottom_canvas = metric_to_canvas(bottom, bounds, bev_size)
        center_xy = metric_to_canvas(box[:2][None, :], bounds, bev_size)[0]
        heading_xy = box_corners[[0, 4], :2].mean(axis=0, keepdims=True)
        heading_canvas = metric_to_canvas(heading_xy, bounds, bev_size)[0]
        velocity = box[7:9] if box.shape[0] >= 9 else np.zeros(2, dtype=np.float32)
        velocity_tip = box[:2] + velocity * 1.5
        velocity_canvas = metric_to_canvas(
            np.array([velocity_tip], dtype=np.float32), bounds, bev_size)[0]

        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, bottom_canvas, color)
        canvas = alpha_blend(canvas, overlay, 0.22)

        for start, end in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(canvas, tuple(bottom_canvas[start]),
                     tuple(bottom_canvas[end]), color, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(center_xy), 3, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(center_xy), tuple(heading_canvas), color,
                        3, cv2.LINE_AA, tipLength=0.25)
        if np.linalg.norm(velocity) > 0.2:
            cv2.arrowedLine(canvas, tuple(center_xy), tuple(velocity_canvas),
                            (245, 245, 245), 2, cv2.LINE_AA, tipLength=0.22)

        legend_counts[label_name] = legend_counts.get(label_name, 0) + 1
        text = f'{label_name} {score:.2f}'
        anchor = tuple(bottom_canvas[0] + np.array([6, -6]))
        draw_text_tag(canvas, text, anchor, color)

    center = metric_to_canvas(np.array([[0.0, 0.0]], dtype=np.float32),
                              bounds, bev_size)[0]
    ego_motion_tip = metric_to_canvas(
        np.array([[ego_motion['motion_xy'][0] * 2.0,
                   ego_motion['motion_xy'][1] * 2.0]], dtype=np.float32),
        bounds, bev_size)[0]
    if ego_motion['speed_mps'] > 0.2:
        cv2.arrowedLine(canvas, tuple(center), tuple(ego_motion_tip),
                        (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.22)

    speed_text = f'ego speed {ego_motion["speed_kph"]:.1f} km/h'
    cv2.putText(canvas, speed_text, (14, bev_size - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2,
                cv2.LINE_AA)
    cv2.putText(canvas,
                f'front:[{forward_min:.0f},{forward_max:.0f}]m left:[{left_min:.0f},{left_max:.0f}]m',
                (14, bev_size - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (210, 210, 210), 1, cv2.LINE_AA)
    legend_y = 58
    for label_name, count in sorted(legend_counts.items(),
                                    key=lambda item: (-item[1], item[0]))[:6]:
        color = CLASS_COLOR_BY_NAME.get(
            label_name, CLASS_COLORS[class_names.index(label_name) %
                                     len(CLASS_COLORS)])
        cv2.rectangle(canvas, (bev_size - 210, legend_y - 14),
                      (bev_size - 194, legend_y + 2), color, -1)
        cv2.putText(canvas, f'{label_name}: {count}', (bev_size - 186, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1,
                    cv2.LINE_AA)
        legend_y += 24
    return draw_title(canvas, 'Top-down BEV Obstacles')


def compose_side_panel(bev_heatmap, bev_boxes):
    panel = bev_boxes.copy()
    inset = cv2.resize(
        bev_heatmap, (panel.shape[1] // 3, panel.shape[0] // 3),
        interpolation=cv2.INTER_AREA)
    x0 = panel.shape[1] - inset.shape[1] - 18
    y0 = panel.shape[0] - inset.shape[0] - 18
    cv2.rectangle(panel, (x0 - 6, y0 - 6),
                  (x0 + inset.shape[1] + 6, y0 + inset.shape[0] + 6),
                  (20, 20, 20), -1)
    cv2.rectangle(panel, (x0 - 6, y0 - 6),
                  (x0 + inset.shape[1] + 6, y0 + inset.shape[0] + 6),
                  (235, 235, 235), 1)
    panel[y0:y0 + inset.shape[0], x0:x0 + inset.shape[1]] = inset
    cv2.putText(panel, 'feature inset', (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def overlay_runtime_info(frame, sample_meta, frame_idx, inference_ms,
                         avg_ms, num_detections, ego_speed_kph):
    footer_height = 92
    footer = np.zeros((footer_height, frame.shape[1], 3), dtype=np.uint8)
    footer[:] = (18, 18, 18)
    lines = [
        f'frame={frame_idx}  sample_index={sample_meta["sample_index"]}  token={sample_meta["sample_token"]}',
        f'inference={inference_ms:.2f} ms  avg={avg_ms:.2f} ms  detections={num_detections}  ego={ego_speed_kph:.1f} km/h',
        f'timestamp={sample_meta.get("timestamp")}'
    ]
    for idx, text in enumerate(lines):
        cv2.putText(footer, text, (18, 28 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return np.concatenate([frame, footer], axis=0)


def resolve_frame_indices(dataset, start_index, start_token, num_frames, stride):
    start_index = resolve_sample_index(dataset, start_index, start_token)
    indices = []
    current = start_index
    while current < len(dataset) and len(indices) < num_frames:
        indices.append(current)
        current += stride
    if not indices:
        raise RuntimeError('No frames resolved for visualization.')
    return indices


def main():
    args = parse_args()
    args.device = normalize_device(args.device)

    os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/.mplconfig').resolve()))
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

    cfg = prepare_cfg(args.config, args.cfg_options)
    setup_multi_processes(cfg)
    dataset = build_dataset(cfg.data.test)
    frame_indices = resolve_frame_indices(dataset, args.start_index,
                                          args.start_token, args.num_frames,
                                          args.stride)

    model = init_model(cfg, checkpoint=args.checkpoint, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / 'frames'
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    bounds = get_bev_bounds(model)
    stats = []
    video_writer = None

    for render_idx, sample_index in enumerate(frame_indices):
        batch = prepare_data(model, dataset, sample_index)
        result, bev_feature, inference_ms = run_inference(
            model, batch, rescale=True, return_bev_feature=True)
        sample_meta = get_sample_meta(dataset, sample_index)
        ego_motion = estimate_ego_motion(dataset, sample_index)
        filtered_boxes, _, _, _ = filter_detection_result(
            result, dataset.CLASSES, args.score_thr, args.topk)
        num_detections = int(filtered_boxes.shape[0])

        stats.append({
            'sample_index': sample_index,
            'sample_token': sample_meta['sample_token'],
            'inference_ms': inference_ms,
            'num_detections': num_detections,
            'ego_speed_kph': ego_motion['speed_kph'],
        })
        valid_times = [
            item['inference_ms'] for item in stats[min(args.warmup, len(stats)):]
        ]
        avg_ms = float(np.mean(valid_times)) if valid_times else inference_ms

        camera_grid = compose_camera_grid(sample_meta, args.cam_width)
        bev_heatmap = bev_feature_to_heatmap(bev_feature, args.bev_size)
        bev_boxes = render_bev_boxes_panel(result, dataset.CLASSES, bounds,
                                           args.bev_size, args.score_thr, args.topk,
                                           ego_motion)
        side_panel = compose_side_panel(bev_heatmap, bev_boxes)
        if side_panel.shape[0] != camera_grid.shape[0]:
            side_panel = cv2.resize(side_panel,
                                    (side_panel.shape[1], camera_grid.shape[0]))
        frame = np.concatenate([camera_grid, side_panel], axis=1)
        frame = overlay_runtime_info(frame, sample_meta, render_idx,
                                     inference_ms, avg_ms, num_detections,
                                     ego_motion['speed_kph'])

        if video_writer is None:
            video_path = out_dir / args.video_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps,
                                           (frame.shape[1], frame.shape[0]))
        video_writer.write(frame)

        if args.save_frames:
            frame_path = frames_dir / f'{render_idx:05d}_{sample_meta["sample_token"]}.jpg'
            cv2.imwrite(str(frame_path), frame)

        print(
            f'frame={render_idx:03d} sample_index={sample_index} '
            f'token={sample_meta["sample_token"]} '
            f'inference_ms={inference_ms:.2f} '
            f'avg_ms={avg_ms:.2f} '
            f'detections={num_detections}')

    if video_writer is not None:
        video_writer.release()

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': args.device,
        'start_index': frame_indices[0],
        'num_frames': len(frame_indices),
        'stride': args.stride,
        'fps': args.fps,
        'warmup': args.warmup,
        'video_path': str((out_dir / args.video_name).resolve()),
        'frames': stats,
    }
    valid_times = [item['inference_ms'] for item in stats[min(args.warmup, len(stats)):]]
    summary['avg_inference_ms'] = float(np.mean(valid_times)) if valid_times else \
        float(np.mean([item['inference_ms'] for item in stats]))
    summary['avg_fps'] = 1000.0 / max(summary['avg_inference_ms'], 1e-6)
    summary_path = out_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'video_path: {out_dir / args.video_name}')
    print(f'summary_json: {summary_path}')
    print(f'avg_inference_ms: {summary["avg_inference_ms"]:.2f}')
    print(f'avg_fps: {summary["avg_fps"]:.2f}')


if __name__ == '__main__':
    main()
