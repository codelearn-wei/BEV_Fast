#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.realtime_perception.frame_sources import (  # noqa: E402
    DirectoryFrameSource,
    FrameSource,
    TCPFrameSource,
)
from projects.realtime_perception.inferencers import (  # noqa: E402
    build_inferencer,
    build_payload,
)
from projects.realtime_perception.visualizer import RuntimeVisualizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Engineering realtime perception pipeline for FastBEV.')
    parser.add_argument('--backend', choices=['pth', 'trt'], default='pth')
    parser.add_argument('--config', required=True, help='model config path')
    parser.add_argument('--checkpoint', default='', help='pth checkpoint path')
    parser.add_argument('--engine', default='', help='trt engine path')
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument(
        '--source', choices=['directory', 'tcp'], default='directory')
    parser.add_argument(
        '--frame-dir',
        default='outputs/realtime_perception/incoming',
        help='directory source: *.json frame packets')
    parser.add_argument('--tcp-host', default='0.0.0.0')
    parser.add_argument('--tcp-port', type=int, default=17999)

    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--topk', type=int, default=120)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--no-save-video', action='store_true')
    parser.add_argument('--out-dir', default='outputs/realtime_perception/run')
    parser.add_argument('--idle-timeout', type=float, default=0.2)
    return parser.parse_args()


def build_source(args: argparse.Namespace) -> FrameSource:
    if args.source == 'directory':
        return DirectoryFrameSource(Path(args.frame_dir), poll_interval_s=0.1)
    return TCPFrameSource(host=args.tcp_host, port=args.tcp_port)


def main() -> None:
    args = parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/.mplconfig').resolve()))
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / 'json'
    json_dir.mkdir(parents=True, exist_ok=True)

    inferencer = build_inferencer(
        backend=args.backend,
        config=args.config,
        checkpoint=args.checkpoint,
        engine=args.engine,
        device=args.device)

    source = build_source(args)
    visualizer = RuntimeVisualizer(
        out_dir=out_dir,
        fps=args.fps,
        show=args.show,
        save_frames=args.save_frames,
        save_video=not args.no_save_video,
        bounds=(-20.0, 70.0, -30.0, 30.0))

    stop_flag = {'stop': False}

    def _handle_signal(signum, _frame):
        _ = signum
        stop_flag['stop'] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print('[realtime] pipeline started')
    print(f'[realtime] backend={args.backend} source={args.source} out_dir={out_dir}')

    frame_idx = 0
    infer_times = []

    try:
        while not stop_flag['stop']:
            packet = source.next_frame(timeout_s=args.idle_timeout)
            if packet is None:
                continue

            result, inference_ms = inferencer.infer(packet)
            payload = build_payload(
                result=result,
                inference_ms=inference_ms,
                class_names=inferencer.class_names,
                frame=packet,
                score_thr=args.score_thr,
                topk=args.topk,
                backend=args.backend)

            infer_times.append(inference_ms)
            avg_ms = sum(infer_times) / float(len(infer_times))
            visualizer.render(packet, payload, avg_ms=avg_ms, frame_idx=frame_idx)

            out_path = json_dir / f'{frame_idx:06d}_{packet.frame_id}.json'
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding='utf-8')

            print(
                f'[realtime] frame={frame_idx:06d} id={packet.frame_id} '
                f'infer={inference_ms:.2f}ms avg={avg_ms:.2f}ms '
                f'dets={payload["num_detections"]}')

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        source.close()
        visualizer.close()

    if infer_times:
        avg_ms = sum(infer_times) / float(len(infer_times))
        print(f'[realtime] completed frames={len(infer_times)} avg_ms={avg_ms:.2f} '
              f'avg_fps={1000.0 / max(avg_ms, 1e-6):.2f}')
    else:
        print('[realtime] no frames processed')


if __name__ == '__main__':
    main()
