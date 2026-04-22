# Local Environment Setup

This repository has been prepared to use a project-local virtual environment at `.venv`.

Recommended interpreter:

- Python 3.8

Quick start:

```bash
source .venv/bin/activate
python --version
python -c "import torch, mmcv; print(torch.__version__, mmcv.__version__)"
```

If the environment needs to be recreated:

```bash
/home/ego_vehicle/miniconda3/envs/ros_carla/bin/python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Primary install flow used for this project:

```bash
pip install openmim
pip install torch==1.10.0 torchvision==0.11.1
mim install "mmcv-full==1.6.0"
pip install mmdet==2.25.1 mmsegmentation==0.25.0
pip install -r requirements/runtime.txt
pip install -v -e .
```

Current machine note:

- The local CUDA toolkit is `12.3`, while the selected PyTorch build is `cu111`.
- To keep the repository installable on this machine, setup supports
  `SKIP_CUDA_EXT=1`, and `bev_pool_v2` falls back to a pure PyTorch
  implementation when the custom extension is unavailable.
- This fallback is suitable for development and functional inference checks,
  but it will be slower than the native CUDA extension.
