#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import compat_cfg

try:
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

from mmdet3d.apis import init_model


def normalize_device(device):
    if device != 'cpu' and not torch.cuda.is_available():
        print('warning: cuda device requested but unavailable, '
              'falling back to cpu')
        return 'cpu'
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description='FastBEV single-sample inference and JSON export')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=None,
        help='sample index in cfg.data.test ann_file')
    parser.add_argument(
        '--sample-token',
        type=str,
        default=None,
        help='sample token in cfg.data.test ann_file')
    parser.add_argument(
        '--device', default='cuda:0', help='device for inference')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold for exported boxes')
    parser.add_argument(
        '--topk',
        type=int,
        default=100,
        help='max number of detections to export')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs/fastbev_infer',
        help='directory for json outputs')
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='explicit json output path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings with key=value pairs')
    return parser.parse_args()


def prepare_cfg(config_path, cfg_options=None):
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    return cfg


def resolve_sample_index(dataset, sample_index=None, sample_token=None):
    if sample_index is not None and sample_token is not None:
        raise ValueError('Only one of --sample-index or --sample-token can be set.')
    if sample_index is None and sample_token is None:
        sample_index = 0

    if sample_token is not None:
        for idx, info in enumerate(dataset.data_infos):
            token = info.get('token', info.get('sample_idx'))
            if token == sample_token:
                return idx
        raise KeyError(f'Sample token not found: {sample_token}')

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f'sample-index out of range: {sample_index}')
    return sample_index


def prepare_data(model, dataset, sample_index):
    data = dataset.prepare_test_data(sample_index)
    if data is None:
        raise RuntimeError(f'Failed to prepare sample {sample_index}')

    batch = collate([data], samples_per_gpu=1)
    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        batch = scatter(batch, [device.index])[0]
    else:
        if 'img_metas' in batch and batch['img_metas']:
            batch['img_metas'] = batch['img_metas'][0].data[0]
        if 'points' in batch and batch['points']:
            batch['points'] = batch['points'][0].data[0]
        if 'img_inputs' in batch and batch['img_inputs']:
            batch['img_inputs'] = batch['img_inputs'][0]

    if 'img_metas' in batch and batch['img_metas'] is not None:
        batch['img_metas'] = [batch['img_metas']]
    if 'points' in batch and batch['points'] is not None:
        batch['points'] = [batch['points']]
    if 'img_inputs' in batch and batch['img_inputs'] is not None:
        batch['img_inputs'] = [batch['img_inputs']]
    return batch


def tensor_to_list(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return value


def get_sample_meta(dataset, sample_index):
    info = dataset.data_infos[sample_index]
    sample_token = info.get('token', info.get('sample_idx'))
    meta = {
        'sample_index': sample_index,
        'sample_token': sample_token,
        'timestamp': info.get('timestamp'),
    }
    if 'cams' in info:
        meta['camera_files'] = {
            cam_name: cam_info['data_path']
            for cam_name, cam_info in info['cams'].items()
        }
    if 'lidar_path' in info:
        meta['lidar_path'] = info['lidar_path']
    return meta


def serialize_result(result, class_names, score_thr, topk):
    det = result['pts_bbox'] if 'pts_bbox' in result else result
    boxes = det['boxes_3d'].tensor.detach().cpu()
    scores = det['scores_3d'].detach().cpu()
    labels = det['labels_3d'].detach().cpu()

    keep = scores >= score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if topk is not None and boxes.shape[0] > topk:
        order = torch.argsort(scores, descending=True)[:topk]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        label_id = int(label.item())
        detections.append({
            'label': label_id,
            'label_name': class_names[label_id],
            'score': float(score.item()),
            'box_3d': [float(x) for x in box.tolist()],
        })
    return detections


def main():
    args = parse_args()
    args.device = normalize_device(args.device)

    os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/.mplconfig').resolve()))
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

    cfg = prepare_cfg(args.config, args.cfg_options)
    dataset = build_dataset(cfg.data.test)
    sample_index = resolve_sample_index(dataset, args.sample_index,
                                        args.sample_token)

    model = init_model(cfg, checkpoint=args.checkpoint, device=args.device)
    batch = prepare_data(model, dataset, sample_index)

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **batch)[0]

    sample_meta = get_sample_meta(dataset, sample_index)
    detections = serialize_result(result, dataset.CLASSES, args.score_thr,
                                  args.topk)

    payload = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': args.device,
        'score_thr': args.score_thr,
        'topk': args.topk,
        'sample': sample_meta,
        'num_detections': len(detections),
        'detections': detections,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        out_path = Path(args.out)
    else:
        sample_name = sample_meta['sample_token'] or f'index_{sample_index}'
        out_path = out_dir / f'{sample_name}.json'

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'sample_index: {sample_index}')
    print(f'sample_token: {sample_meta["sample_token"]}')
    print(f'num_detections: {len(detections)}')
    print(f'output_json: {out_path}')


if __name__ == '__main__':
    main()
