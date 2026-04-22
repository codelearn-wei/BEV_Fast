#!/usr/bin/env bash

export CONDA_ENVS_PATH=/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/.conda_envs
export MPLCONFIGDIR=/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/.cache/matplotlib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
mkdir -p "$MPLCONFIGDIR"

conda activate bev
