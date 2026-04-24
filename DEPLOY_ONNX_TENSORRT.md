# FastBEV ONNX / TensorRT 部署说明

这份文档面向当前仓库 `advanced-fastbev-fastbev`，目标是把已经跑通的 PyTorch 推理链路继续推进到:

1. 导出 ONNX
2. 生成 TensorRT engine
3. 对比 PyTorch 与 TensorRT 的时延

## 1. 当前仓库里的相关脚本

- ONNX 导出脚本: [tools/convert_fastbev_to_TRT.py](/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/tools/convert_fastbev_to_TRT.py)
- TensorRT benchmark 脚本: [tools/analysis_tools/benchmark_trt_fastbev.py](/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/tools/analysis_tools/benchmark_trt_fastbev.py)
- 部署环境检查脚本: [tools/check_deploy_env.py](/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/tools/check_deploy_env.py)
- ONNX 基础依赖列表: [requirements/deploy.txt](/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/requirements/deploy.txt)

## 2. 建议的部署顺序

### 第一步：检查环境

```bash
source /home/ego_vehicle/miniconda3/etc/profile.d/conda.sh
source ./activate_bev.sh
python tools/check_deploy_env.py
```

理想状态:

- `onnx` 已安装
- `onnxsim` 已安装
- `tensorrt` 已安装
- `mmdeploy` 已安装
- `nvidia-smi` 正常

## 3. 安装 ONNX 基础依赖

先安装 ONNX 相关的 Python 包:

```bash
pip install -r requirements/deploy.txt
```

等价命令:

```bash
pip install onnx==1.13.1 onnxsim==0.4.36
```

安装完成后再次检查:

```bash
python tools/check_deploy_env.py
```

## 4. 导出 ONNX

建议先创建导出目录:

```bash
mkdir -p outputs/deploy
```

然后执行导出:

```bash
python tools/convert_fastbev_to_TRT.py \
  configs/fastbev/paper/fastbev-r50-cbgs.py \
  model/fastbev-r50-cbgs.pth \
  outputs/deploy/
```

正常情况下，导出脚本会:

- 构建测试集和模型
- 从一帧样本中提取 ONNX 导出所需输入
- 导出 `.onnx`
- 自动执行 `onnx.checker`
- 自动执行 `onnxsim.simplify`

导出结果通常在:

```bash
outputs/deploy/fastbev.onnx
```

如果你打开导出脚本，会看到它实际导出的输入是:

- `img`
- `coors_img`
- `coors_depth`

这说明当前导出的 ONNX 是 FastBEV 的部署子图，不是把整个数据预处理流水线都打包进 ONNX。

## 5. TensorRT 安装原则

这一部分最重要的不是“命令”，而是“版本匹配”。

你需要确保下面几件事同时匹配:

- Python 版本
- CUDA 版本
- TensorRT runtime 版本
- TensorRT Python binding 版本

在你的 WSL 环境里，PyTorch 当前是 `cu111`，但 Windows 驱动更高版本通常没问题。真正要注意的是:

- TensorRT Python wheel 要和 TensorRT runtime 对应
- 不要混装来源不一致的 TensorRT

### 推荐做法

1. 先确认系统里有没有 TensorRT runtime
2. 再安装与其匹配的 Python binding
3. 最后补 `mmdeploy`

## 6. 生成 TensorRT engine

当前仓库的 `tools/convert_fastbev_to_TRT.py` 主要负责导出 ONNX。  
engine 生成依赖 TensorRT / mmdeploy 环境完整可用后再进行。

常见路径有两种:

### 路径 A：使用 `trtexec`

如果系统里已有 `trtexec`，可以直接从 ONNX 生成 engine:

```bash
trtexec \
  --onnx=outputs/deploy/fastbev.onnx \
  --saveEngine=outputs/deploy/fastbev_fp16.engine \
  --fp16
```

如果想先保守验证，也可以先不用 FP16:

```bash
trtexec \
  --onnx=outputs/deploy/fastbev.onnx \
  --saveEngine=outputs/deploy/fastbev_fp32.engine
```

### 路径 B：使用 mmdeploy / Python 接口

如果你后面想把整个部署流程脚本化，或者继续接更复杂的动态 shape / calibrate / INT8，就需要把 `mmdeploy` 和 TensorRT Python 环境补齐。

## 7. TensorRT benchmark

有了 engine 后，使用仓库自带脚本测速:

```bash
python tools/analysis_tools/benchmark_trt_fastbev.py 
  configs/fastbev/paper/fastbev-r50-cbgs.py 
  outputs/deploy/fastbev_fp16.engine 
  --samples 200
```

这个脚本会:

- 加载 TensorRT engine
- 逐帧执行推理
- 打印 FPS
- 打印平均单帧时延

## 8. 如何与当前 PyTorch 基线对比

当前仓库里已经有两个基线来源:

### 单帧

```bash
python tools/fastbev_infer.py 
  configs/fastbev/paper/fastbev-r50-cbgs.py 
  model/fastbev-r50-cbgs.pth 
  --sample-index 10 
  --device cuda:0
```

输出里会包含:

```bash
inference_ms: xxx
```

### 连续帧

```bash
python tools/fastbev_video_demo.py 
  configs/fastbev/paper/fastbev-r50-cbgs.py 
  model/fastbev-r50-cbgs.pth 
  --start-index 10 
  --num-frames 30 
  --device cuda:0
```

输出目录中的 `summary.json` 会包含:

- `avg_inference_ms`
- `avg_fps`

因此建议对比方式是:

1. 先记录 `summary.json` 的 PyTorch 基线
2. 再跑 `benchmark_trt_fastbev.py`
3. 比较平均时延和 FPS

## 9. 常见问题

### 问题 1：`onnx` / `onnxsim` 缺失

直接安装:

```bash
pip install -r requirements/deploy.txt
```

### 问题 2：`No module named tensorrt`

说明 TensorRT Python binding 还没装好，不是仓库代码问题。

### 问题 3：`trtexec` 找不到

说明 TensorRT runtime / binary 工具没装到 PATH 中。

### 问题 4：WSL 里 `nvidia-smi` 异常

先确认:

```bash
/usr/lib/wsl/lib/nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

并确保 `activate_bev.sh` 已经执行，因为里面补了:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## 10. 建议的实际推进顺序

如果你下一步准备亲自安装和部署，我建议就按这个顺序来:

1. 跑 `python tools/check_deploy_env.py`
2. 安装 `onnx` 和 `onnxsim`
3. 先把 ONNX 成功导出
4. 再安装 TensorRT runtime / Python binding
5. 用 `trtexec` 生成 engine
6. 用 `benchmark_trt_fastbev.py` 测速
7. 和 `summary.json` 做对比

这样最稳，也最容易定位问题。  
