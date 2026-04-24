from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.loading import PrepareImageInputs

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fastbev_infer import normalize_device, prepare_cfg, serialize_result
from tools.fastbev_trt_runtime import FastBEVTRTInferencer
from mmdet3d.apis import init_model

from .schemas import CAMERA_ORDER, FramePacket


class BaseInferencer(ABC):

    @property
    @abstractmethod
    def class_names(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, frame: FramePacket) -> Tuple[Dict[str, Any], float]:
        raise NotImplementedError


class PthFastBEVInferencer(BaseInferencer):

    def __init__(self, config: str, checkpoint: str, device: str) -> None:
        self.device_str = normalize_device(device)
        self.cfg = prepare_cfg(config)
        self.dataset = build_dataset(self.cfg.data.test)
        self._class_names = self.dataset.CLASSES
        self.image_preparer = PrepareImageInputs(
            self.cfg.data_config, is_train=False, sequential=False)
        self.model = init_model(
            self.cfg, checkpoint=checkpoint, device=self.device_str)
        self.device = next(self.model.parameters()).device

    @property
    def class_names(self):
        return self._class_names

    @staticmethod
    def _quat_trans_to_mat(rotation, translation):
        w, x, y, z = rotation
        rot = torch.tensor(
            Quaternion(w, x, y, z).rotation_matrix, dtype=torch.float32)
        trans = torch.tensor(translation, dtype=torch.float32)
        matrix = torch.eye(4, dtype=torch.float32)
        matrix[:3, :3] = rot
        matrix[:3, 3] = trans
        return matrix

    @staticmethod
    def _to_pil(image_or_path, color_order='bgr'):
        if isinstance(image_or_path, (str, Path)):
            return Image.open(str(image_or_path))
        image = np.asarray(image_or_path)
        if color_order.lower() == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image.astype(np.uint8))

    def _build_img_inputs(self, frame: FramePacket):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        bda = torch.eye(4, dtype=torch.float32)

        resize = resize_dims = crop = flip = rotate = None
        for cam_name in CAMERA_ORDER:
            image = self._to_pil(frame.camera_images[cam_name], frame.image_color)
            info = frame.camera_infos[cam_name]

            post_rot = torch.eye(2, dtype=torch.float32)
            post_tran = torch.zeros(2, dtype=torch.float32)

            intrin = torch.tensor(info['cam_intrinsic'], dtype=torch.float32)
            sensor2ego = self._quat_trans_to_mat(
                info['sensor2ego_rotation'], info['sensor2ego_translation'])
            ego2global = self._quat_trans_to_mat(
                frame.ego2global_rotation, frame.ego2global_translation)

            if resize is None:
                resize, resize_dims, crop, flip, rotate = \
                    self.image_preparer.sample_augmentation(
                        H=image.height, W=image.width, flip=False, scale=None)

            image, post_rot2, post_tran2 = self.image_preparer.img_transform(
                image, post_rot, post_tran, resize, resize_dims, crop, flip,
                rotate)

            post_rot_full = torch.eye(3, dtype=torch.float32)
            post_tran_full = torch.zeros(3, dtype=torch.float32)
            post_rot_full[:2, :2] = post_rot2
            post_tran_full[:2] = post_tran2

            imgs.append(self.image_preparer.normalize_img(image))
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            intrins.append(intrin)
            post_rots.append(post_rot_full)
            post_trans.append(post_tran_full)

        return (
            torch.stack(imgs).unsqueeze(0),
            torch.stack(sensor2egos).unsqueeze(0),
            torch.stack(ego2globals).unsqueeze(0),
            torch.stack(intrins).unsqueeze(0),
            torch.stack(post_rots).unsqueeze(0),
            torch.stack(post_trans).unsqueeze(0),
            bda.unsqueeze(0),
        )

    def infer(self, frame: FramePacket) -> Tuple[Dict[str, Any], float]:
        img_inputs = [item.to(self.device) for item in self._build_img_inputs(frame)]
        img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]

        is_cuda = self.device.type == 'cuda'
        if is_cuda:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        with torch.no_grad():
            img_feats, _, _ = self.model.extract_feat(
                None, img=img_inputs, img_metas=img_metas)
            bbox_pts = self.model.simple_test_pts(
                img_feats, img_metas, rescale=True)

        if is_cuda:
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - start) * 1000.0
        return {'pts_bbox': bbox_pts[0]}, inference_ms


class TrtFastBEVInferencer(BaseInferencer):

    def __init__(self,
                 config: str,
                 engine: str,
                 checkpoint: str,
                 device: str) -> None:
        self.trt = FastBEVTRTInferencer(
            config=config,
            engine=engine,
            checkpoint=checkpoint,
            device=device,
            cfg_options=None)

    @property
    def class_names(self):
        return self.trt.dataset.CLASSES

    def infer(self, frame: FramePacket) -> Tuple[Dict[str, Any], float]:
        result, inference_ms, _ = self.trt.infer_custom_images(
            camera_images=frame.camera_images,
            camera_infos=frame.camera_infos,
            ego2global_rotation=frame.ego2global_rotation,
            ego2global_translation=frame.ego2global_translation,
            cache_key=frame.metadata.get('cache_key', 'realtime_rig'),
            image_color=frame.image_color)
        return result, inference_ms


def build_inferencer(backend: str,
                     config: str,
                     checkpoint: str,
                     device: str,
                     engine: str = '') -> BaseInferencer:
    backend = backend.lower().strip()
    if backend == 'pth':
        if not checkpoint:
            raise ValueError('checkpoint is required for pth backend.')
        return PthFastBEVInferencer(
            config=config,
            checkpoint=checkpoint,
            device=device)

    if backend == 'trt':
        if not engine:
            raise ValueError('engine is required for trt backend.')
        return TrtFastBEVInferencer(
            config=config,
            engine=engine,
            checkpoint=checkpoint,
            device=device)

    raise ValueError(f'Unsupported backend: {backend}')


def build_payload(result: Dict[str, Any],
                  inference_ms: float,
                  class_names,
                  frame: FramePacket,
                  score_thr: float,
                  topk: int,
                  backend: str) -> Dict[str, Any]:
    detections = serialize_result(
        result=result,
        class_names=class_names,
        score_thr=score_thr,
        topk=topk)
    return {
        'backend': backend,
        'frame_id': frame.frame_id,
        'timestamp': frame.timestamp,
        'inference_ms': inference_ms,
        'score_thr': score_thr,
        'topk': topk,
        'num_detections': len(detections),
        'sample': frame.to_infer_metadata(),
        'detections': detections,
    }
