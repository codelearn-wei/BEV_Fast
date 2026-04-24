#!/usr/bin/env python3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image
from mmcv.runner import load_checkpoint
from pyquaternion import Quaternion

from trt_env import bootstrap_tensorrt_import
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.loading import PrepareImageInputs
from mmdet3d.models import build_model

from fastbev_infer import (get_sample_meta, normalize_device, prepare_cfg,
                           prepare_data, serialize_result)


def torch_dtype_from_trt(dtype) -> torch.dtype:
    bootstrap_tensorrt_import()
    import tensorrt as trt
    if dtype == trt.bool:
        return torch.bool
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    raise TypeError(f'{dtype} is not supported by torch')


class TRTWrapper(torch.nn.Module):

    def __init__(self,
                 engine: Union[str, Any],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        bootstrap_tensorrt_import()
        import tensorrt as trt
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def get_input_profile_shape(self, input_name: str):
        idx = self.engine.get_binding_index(input_name)
        try:
            return self.engine.get_profile_shape(0, idx)
        except TypeError:
            return self.engine.get_profile_shape(0, input_name)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)
            engine_shape = tuple(self.engine.get_binding_shape(idx))
            if len(engine_shape) != input_tensor.dim():
                raise ValueError(
                    f'TensorRT binding rank mismatch for {input_name}: '
                    f'engine expects rank {len(engine_shape)} with shape '
                    f'{engine_shape}, got tensor shape {tuple(input_tensor.shape)}')
            input_dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if input_tensor.dtype != input_dtype:
                input_tensor = input_tensor.to(dtype=input_dtype)
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            output = torch.zeros(size=shape, dtype=dtype, device='cuda')
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream)
        return outputs


class FastBEVTRTInferencer:

    def __init__(self,
                 config: str,
                 engine: str,
                 checkpoint: Optional[str] = None,
                 device: str = 'cuda:0',
                 cfg_options: Optional[Dict] = None) -> None:
        device = normalize_device(device)
        if device == 'cpu':
            raise RuntimeError('TensorRT inference requires a CUDA device.')

        bootstrap_tensorrt_import()
        from mmdeploy.backend.tensorrt import load_tensorrt_plugin
        load_tensorrt_plugin()

        self.cfg = prepare_cfg(config, cfg_options)
        self.cfg.model.pretrained = None
        self.cfg.model.type = self.cfg.model.type + 'TRT'
        self.cfg.model.img_view_transformer.accelerate = True
        self.device = torch.device(device)
        self.dataset = build_dataset(self.cfg.data.test)
        self.image_preparer = PrepareImageInputs(
            self.cfg.data_config, is_train=False, sequential=False)

        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        if checkpoint is not None:
            load_checkpoint(self.model, checkpoint, map_location='cpu')
        self.model.to(self.device)
        self.model.eval()

        output_names = [
            f'output_{i}'
            for i in range(6 * len(self.model.pts_bbox_head.task_heads))
        ]
        self.trt_model = TRTWrapper(engine, output_names)
        self.coors_cache = {}

    def _pad_tensor_to_shape(self, tensor, target_shape, pad_value=0):
        if tuple(tensor.shape) == tuple(target_shape):
            return tensor.contiguous()
        if tensor.dim() != len(target_shape):
            raise ValueError(
                f'Cannot pad tensor of shape {tuple(tensor.shape)} '
                f'to target shape {tuple(target_shape)}')
        if tensor.dim() != 1:
            raise ValueError(
                f'Only 1D dynamic tensors are supported for padding, '
                f'got shape {tuple(tensor.shape)}')
        if tensor.shape[0] > target_shape[0]:
            raise ValueError(
                f'Tensor length {tensor.shape[0]} exceeds engine profile '
                f'limit {target_shape[0]}. Please rebuild the engine with a '
                f'larger shape profile.')
        padded = torch.full(
            target_shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device)
        padded[:tensor.shape[0]] = tensor
        return padded.contiguous()

    def _align_engine_inputs(self, engine_inputs):
        aligned_inputs = {}
        for input_name, tensor in engine_inputs.items():
            engine_shape = tuple(
                self.trt_model.engine.get_binding_shape(
                    self.trt_model.engine.get_binding_index(input_name)))
            if -1 in engine_shape:
                _, _, max_shape = self.trt_model.get_input_profile_shape(
                    input_name)
                target_shape = tuple(max_shape)
                if tensor.dim() == len(target_shape) and tuple(
                        tensor.shape) != target_shape:
                    tensor = self._pad_tensor_to_shape(
                        tensor, target_shape, pad_value=0)
            aligned_inputs[input_name] = tensor
        return aligned_inputs

    def make_bda_identity(self):
        bda = torch.eye(4, dtype=torch.float32)
        return bda

    def _quat_trans_to_mat(self, rotation, translation):
        w, x, y, z = rotation
        rot = torch.tensor(
            Quaternion(w, x, y, z).rotation_matrix, dtype=torch.float32)
        trans = torch.tensor(translation, dtype=torch.float32)
        matrix = torch.eye(4, dtype=torch.float32)
        matrix[:3, :3] = rot
        matrix[:3, 3] = trans
        return matrix

    def _load_image(self, image_or_path, color_order='bgr'):
        if isinstance(image_or_path, (str, Path)):
            image = Image.open(str(image_or_path))
            return image
        image = np.asarray(image_or_path)
        if color_order.lower() == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image.astype(np.uint8))

    def _ensure_rank(self, tensor, expected_rank, tensor_name):
        while tensor.dim() < expected_rank:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != expected_rank:
            raise ValueError(
                f'Expected {tensor_name} to have rank <= {expected_rank}, '
                f'got {tensor.dim()}')
        return tensor

    def build_custom_img_inputs(self,
                                camera_images: Dict[str, Union[str, np.ndarray]],
                                camera_infos: Dict[str, Dict],
                                ego2global_rotation=None,
                                ego2global_translation=None,
                                cache_key: Optional[str] = None,
                                image_color: str = 'bgr'):
        cam_names = self.cfg.data_config['cams']
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        bda = self.make_bda_identity()

        resize = resize_dims = crop = flip = rotate = None
        for cam_name in cam_names:
            if cam_name not in camera_images:
                raise KeyError(f'Missing image for camera: {cam_name}')
            if cam_name not in camera_infos:
                raise KeyError(f'Missing camera info for camera: {cam_name}')

            image = self._load_image(camera_images[cam_name], image_color)
            info = camera_infos[cam_name]
            post_rot = torch.eye(2, dtype=torch.float32)
            post_tran = torch.zeros(2, dtype=torch.float32)

            intrin = torch.tensor(info['cam_intrinsic'], dtype=torch.float32)
            sensor2ego = self._quat_trans_to_mat(
                info['sensor2ego_rotation'], info['sensor2ego_translation'])

            if 'ego2global_rotation' in info and 'ego2global_translation' in info:
                ego2global = self._quat_trans_to_mat(
                    info['ego2global_rotation'], info['ego2global_translation'])
            elif ego2global_rotation is not None and ego2global_translation is not None:
                ego2global = self._quat_trans_to_mat(
                    ego2global_rotation, ego2global_translation)
            else:
                ego2global = torch.eye(4, dtype=torch.float32)

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

        img_inputs = (
            torch.stack(imgs),
            torch.stack(sensor2egos),
            torch.stack(ego2globals),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
            bda,
        )
        metadata = {
            'camera_files': {
                cam_name: str(camera_images[cam_name])
                if isinstance(camera_images[cam_name], (str, Path))
                else f'in_memory:{cam_name}'
                for cam_name in cam_names
            },
            'cache_key': cache_key,
        }
        return img_inputs, metadata

    def prepare_engine_inputs_from_img_inputs(self, img_inputs, cache_key=None):
        imgs = self._ensure_rank(img_inputs[0], 5, 'img').to(self.device)
        cached = self.coors_cache.get(cache_key) if cache_key is not None else None

        if cached is None:
            with torch.no_grad():
                prepared_inputs = [imgs]
                expected_ranks = [4, 4, 4, 4, 3, 3]
                tensor_names = [
                    'sensor2ego',
                    'ego2global',
                    'cam2img',
                    'post_rots',
                    'post_trans',
                    'bda',
                ]
                for tensor, expected_rank, tensor_name in zip(
                        img_inputs[1:], expected_ranks, tensor_names):
                    prepared_inputs.append(
                        self._ensure_rank(tensor, expected_rank,
                                          tensor_name).to(self.device))
                prepared_inputs = self.model.prepare_inputs(prepared_inputs)
                img_feat, _ = self.model.image_encoder(prepared_inputs[0])
                input_list = [img_feat] + list(prepared_inputs[1:7])
                _, coors_img_list, coors_depth_list = \
                    self.model.img_view_transformer.get_fastray_input(input_list)
                coors_img = coors_img_list[0].contiguous()
                coors_depth = coors_depth_list[0].contiguous()
            if cache_key is not None:
                self.coors_cache[cache_key] = (coors_img, coors_depth)
        else:
            coors_img, coors_depth = cached

        return dict(
            img=imgs[0].contiguous(),
            coors_img=coors_img,
            coors_depth=coors_depth,
        )

    def run_engine(self, engine_inputs):
        torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        outputs = self.trt_model.forward(engine_inputs)
        torch.cuda.synchronize(self.device)
        return outputs, (time.perf_counter() - start_time) * 1000.0

    def postprocess_outputs(self, outputs):
        outputs = [
            outputs[f'output_{i}']
            for i in range(6 * len(self.model.pts_bbox_head.task_heads))
        ]
        pred = self.model.result_deserialize(outputs)
        img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = self.model.pts_bbox_head.get_bboxes(
            pred, img_metas, rescale=True)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return {'pts_bbox': bbox_results[0]}

    def infer_img_inputs(self, img_inputs, cache_key=None):
        engine_inputs = self.prepare_engine_inputs_from_img_inputs(
            img_inputs, cache_key=cache_key)
        engine_inputs = self._align_engine_inputs(engine_inputs)
        outputs, inference_ms = self.run_engine(engine_inputs)
        result = self.postprocess_outputs(outputs)
        return result, inference_ms

    def infer_dataset_sample(self, sample_index):
        batch = prepare_data(self.model, self.dataset, sample_index)
        img_inputs = batch['img_inputs'][0]
        result, inference_ms = self.infer_img_inputs(img_inputs)
        sample_meta = get_sample_meta(self.dataset, sample_index)
        return result, inference_ms, sample_meta

    def infer_custom_images(self,
                            camera_images,
                            camera_infos,
                            ego2global_rotation=None,
                            ego2global_translation=None,
                            cache_key='custom_rig',
                            image_color='bgr'):
        img_inputs, metadata = self.build_custom_img_inputs(
            camera_images=camera_images,
            camera_infos=camera_infos,
            ego2global_rotation=ego2global_rotation,
            ego2global_translation=ego2global_translation,
            cache_key=cache_key,
            image_color=image_color)
        result, inference_ms = self.infer_img_inputs(
            img_inputs, cache_key=cache_key)
        return result, inference_ms, metadata

    def result_to_payload(self,
                          result,
                          inference_ms,
                          sample_meta=None,
                          score_thr=0.3,
                          topk=100):
        detections = serialize_result(
            result, self.dataset.CLASSES, score_thr=score_thr, topk=topk)
        payload = {
            'config': self.cfg.filename,
            'engine_inference_ms': inference_ms,
            'score_thr': score_thr,
            'topk': topk,
            'num_detections': len(detections),
            'detections': detections,
        }
        if sample_meta is not None:
            payload['sample'] = sample_meta
        return payload
