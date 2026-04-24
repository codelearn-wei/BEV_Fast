#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from mmcv import DictAction

from fastbev_infer import get_sample_meta
from fastbev_trt_runtime import FastBEVTRTInferencer
from fastbev_video_demo import (compose_camera_grid, estimate_ego_motion,
                                filter_detection_result,
                                get_bev_bounds, overlay_runtime_info,
                                render_bev_boxes_panel)


def parse_args():
    parser = argparse.ArgumentParser(
        description='FastBEV TensorRT video visualization')
    parser.add_argument('config', help='config file path')
    parser.add_argument('engine', help='TensorRT engine path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='optional checkpoint for helper model construction')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--bev-size', type=int, default=960)
    parser.add_argument('--cam-width', type=int, default=640)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--out-dir', default='outputs/fastbev_trt_demo')
    parser.add_argument('--video-name', default='fastbev_trt_demo.mp4')
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings with key=value pairs')
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = FastBEVTRTInferencer(
        config=args.config,
        engine=args.engine,
        checkpoint=args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)

    dataset = inferencer.dataset
    bounds = get_bev_bounds(inferencer.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / 'frames'
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    video_writer = None
    frame_indices = list(
        range(args.start_index,
              min(len(dataset), args.start_index + args.num_frames * args.stride),
              args.stride))

    for render_idx, sample_index in enumerate(frame_indices):
        result, inference_ms, sample_meta = inferencer.infer_dataset_sample(
            sample_index)
        ego_motion = estimate_ego_motion(dataset, sample_index)
        filtered_boxes, _, _, _ = filter_detection_result(
            result, dataset.CLASSES, args.score_thr, args.topk)
        num_detections = int(filtered_boxes.shape[0])
        stats.append({
            'sample_index': sample_index,
            'sample_token': sample_meta['sample_token'],
            'engine_inference_ms': inference_ms,
            'num_detections': num_detections,
            'ego_speed_kph': ego_motion['speed_kph'],
        })

        valid_times = [
            item['engine_inference_ms']
            for item in stats[min(args.warmup, len(stats)):]
        ]
        avg_ms = float(np.mean(valid_times)) if valid_times else inference_ms

        camera_grid = compose_camera_grid(sample_meta, args.cam_width)
        bev_boxes = render_bev_boxes_panel(
            result=result,
            class_names=dataset.CLASSES,
            bounds=bounds,
            bev_size=args.bev_size,
            score_thr=args.score_thr,
            topk=args.topk,
            ego_motion=ego_motion)

        if bev_boxes.shape[0] != camera_grid.shape[0]:
            bev_boxes = cv2.resize(bev_boxes,
                                   (bev_boxes.shape[1], camera_grid.shape[0]))
        frame = np.concatenate([camera_grid, bev_boxes], axis=1)
        frame = overlay_runtime_info(
            frame=frame,
            sample_meta=sample_meta,
            frame_idx=render_idx,
            inference_ms=inference_ms,
            avg_ms=avg_ms,
            num_detections=num_detections,
            ego_speed_kph=ego_motion['speed_kph'])

        if video_writer is None:
            video_path = out_dir / args.video_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, args.fps,
                (frame.shape[1], frame.shape[0]))
        video_writer.write(frame)

        if args.save_frames:
            frame_path = frames_dir / f'{render_idx:05d}_{sample_meta["sample_token"]}.jpg'
            cv2.imwrite(str(frame_path), frame)

        print(
            f'frame={render_idx:03d} sample_index={sample_index} '
            f'token={sample_meta["sample_token"]} '
            f'engine_ms={inference_ms:.2f} avg_ms={avg_ms:.2f} '
            f'detections={num_detections}')

    if video_writer is not None:
        video_writer.release()

    summary = {
        'config': args.config,
        'engine': args.engine,
        'device': args.device,
        'start_index': frame_indices[0] if frame_indices else args.start_index,
        'num_frames': len(frame_indices),
        'stride': args.stride,
        'fps': args.fps,
        'warmup': args.warmup,
        'video_path': str((out_dir / args.video_name).resolve()),
        'frames': stats,
    }
    valid_times = [
        item['engine_inference_ms']
        for item in stats[min(args.warmup, len(stats)):]
    ]
    summary['avg_engine_ms'] = float(np.mean(valid_times)) if valid_times else \
        float(np.mean([item['engine_inference_ms'] for item in stats]))
    summary['avg_fps'] = 1000.0 / max(summary['avg_engine_ms'], 1e-6)
    summary_path = out_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'video_path: {out_dir / args.video_name}')
    print(f'summary_json: {summary_path}')
    print(f'avg_engine_ms: {summary["avg_engine_ms"]:.2f}')
    print(f'avg_fps: {summary["avg_fps"]:.2f}')


if __name__ == '__main__':
    main()
