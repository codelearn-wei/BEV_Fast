import ctypes
import os
import sys
from pathlib import Path
from typing import Iterable, List


def _existing_dirs(paths: Iterable[str]) -> List[str]:
    seen = set()
    existing = []
    for path in paths:
        if not path:
            continue
        resolved = os.path.realpath(path)
        if resolved in seen or not os.path.isdir(resolved):
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def _candidate_library_dirs() -> List[str]:
    prefixes = [
        sys.prefix,
        os.environ.get('CONDA_PREFIX'),
        os.environ.get('CUDA_HOME'),
        '/usr/local/cuda',
        str(Path.home() / 'miniconda3'),
        str(Path.home() / 'TensorRT-8.6.1'),
    ]
    candidates = []
    for prefix in prefixes:
        if not prefix:
            continue
        candidates.extend([
            os.path.join(prefix, 'lib'),
            os.path.join(prefix, 'lib64'),
            os.path.join(prefix, 'targets', 'x86_64-linux', 'lib'),
            os.path.join(prefix, 'targets', 'x86_64-linux-gnu', 'lib'),
        ])
    candidates.append('/usr/lib/wsl/lib')
    return _existing_dirs(candidates)


def bootstrap_tensorrt_import() -> None:
    """Preload CUDA/TensorRT shared libraries from common local locations."""
    lib_dirs = _candidate_library_dirs()
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    current_dirs = [path for path in current_ld_path.split(':') if path]
    merged_dirs = _existing_dirs([*lib_dirs, *current_dirs])
    os.environ['LD_LIBRARY_PATH'] = ':'.join(merged_dirs)

    required_libs = (
        'libcublas.so.11',
        'libcublasLt.so.11',
        'libcudnn.so.8',
        'libnvinfer.so',
        'libnvinfer_plugin.so',
    )
    for lib_name in required_libs:
        for lib_dir in merged_dirs:
            lib_path = os.path.join(lib_dir, lib_name)
            if not os.path.exists(lib_path):
                continue
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            break
