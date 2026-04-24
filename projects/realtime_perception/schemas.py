from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

CAMERA_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]


@dataclass
class FramePacket:
    frame_id: str
    timestamp: Optional[float]
    camera_images: Dict[str, str]
    camera_infos: Dict[str, Dict[str, Any]]
    ego2global_rotation: List[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    ego2global_translation: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    image_color: str = 'bgr'
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> 'FramePacket':
        missing = [name for name in CAMERA_ORDER
                   if name not in payload.get('camera_images', {})]
        if missing:
            raise KeyError(
                f'Missing camera_images entries for: {", ".join(missing)}')

        missing_info = [name for name in CAMERA_ORDER
                        if name not in payload.get('camera_infos', {})]
        if missing_info:
            raise KeyError(
                f'Missing camera_infos entries for: {", ".join(missing_info)}')

        frame_id = str(payload.get('frame_id', payload.get('sample_token', 'unknown')))
        timestamp = payload.get('timestamp')
        metadata = dict(payload.get('metadata', {}))
        if 'sample_token' not in metadata:
            metadata['sample_token'] = frame_id

        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            camera_images={k: str(v) for k, v in payload['camera_images'].items()},
            camera_infos=dict(payload['camera_infos']),
            ego2global_rotation=list(payload.get(
                'ego2global_rotation', [1.0, 0.0, 0.0, 0.0])),
            ego2global_translation=list(payload.get(
                'ego2global_translation', [0.0, 0.0, 0.0])),
            image_color=str(payload.get('image_color', 'bgr')).lower(),
            metadata=metadata,
        )

    @classmethod
    def from_json_file(cls, json_path: Path) -> 'FramePacket':
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        if 'frame_id' not in payload:
            payload['frame_id'] = json_path.stem
        return cls.from_mapping(payload)

    def to_infer_metadata(self) -> Dict[str, Any]:
        data = dict(self.metadata)
        data.setdefault('frame_id', self.frame_id)
        data.setdefault('timestamp', self.timestamp)
        data.setdefault('camera_files', self.camera_images)
        return data
