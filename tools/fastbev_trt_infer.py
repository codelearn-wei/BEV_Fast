#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from mmcv import DictAction

from fastbev_trt_runtime import FastBEVTRTInferencer


def parse_args():
    parser = argparse.ArgumentParser(
        description='FastBEV TensorRT inference for dataset samples or custom input')
    parser.add_argument('config', help='config file path')
    parser.add_argument('engine', help='TensorRT engine path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='optional checkpoint for helper model construction')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='dataset sample index for inference')
    parser.add_argument(
        '--custom-input',
        type=str,
        default=None,
        help='json file describing custom multi-camera images and calibration')
    parser.add_argument('--device', default='cuda:0', help='inference device')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--out-dir', default='outputs/fastbev_trt_infer')
    parser.add_argument('--out', default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings with key=value pairs')
    return parser.parse_args()


def load_custom_spec(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    camera_images = spec['camera_images']
    camera_infos = spec['camera_infos']
    ego2global_rotation = spec.get('ego2global_rotation')
    ego2global_translation = spec.get('ego2global_translation')
    cache_key = spec.get('cache_key', 'custom_rig')
    image_color = spec.get('image_color', 'bgr')
    metadata = spec.get('metadata', {})
    return (camera_images, camera_infos, ego2global_rotation,
            ego2global_translation, cache_key, image_color, metadata)


def main():
    args = parse_args()
    inferencer = FastBEVTRTInferencer(
        config=args.config,
        engine=args.engine,
        checkpoint=args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)

    sample_meta = None
    if args.custom_input is None:
        result, inference_ms, sample_meta = inferencer.infer_dataset_sample(
            args.sample_index)
    else:
        (camera_images, camera_infos, ego2global_rotation,
         ego2global_translation, cache_key, image_color,
         metadata) = load_custom_spec(args.custom_input)
        result, inference_ms, custom_meta = inferencer.infer_custom_images(
            camera_images=camera_images,
            camera_infos=camera_infos,
            ego2global_rotation=ego2global_rotation,
            ego2global_translation=ego2global_translation,
            cache_key=cache_key,
            image_color=image_color)
        sample_meta = metadata or custom_meta

    payload = inferencer.result_to_payload(
        result=result,
        inference_ms=inference_ms,
        sample_meta=sample_meta,
        score_thr=args.score_thr,
        topk=args.topk)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        out_path = Path(args.out)
    else:
        sample_name = None
        if sample_meta is not None:
            sample_name = sample_meta.get('sample_token')
        if not sample_name:
            sample_name = 'custom_input'
        out_path = out_dir / f'{sample_name}.json'

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'engine_inference_ms: {inference_ms:.2f}')
    print(f'num_detections: {payload["num_detections"]}')
    print(f'output_json: {out_path}')


if __name__ == '__main__':
    main()
