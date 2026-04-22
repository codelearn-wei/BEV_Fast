# FastBEV++ 中文说明

[FastBEV++: Fast by Algorithm, Deployable by Design](https://arxiv.org/abs/2512.08237v1)

本仓库当前维护重点已经调整为:

- 便于继续做 `FastBEV` 推理开发
- 便于提取结构化感知结果
- README 持续保持“能直接照着用”的状态

![](./resources/fastbev_vs_fastvec++_v2.0.png)

## 1. 当前状态

已经完成的基础准备:

- 已清理项目中的 `Zone.Identifier` 垃圾文件
- 已建立项目内的 `conda` 环境 `bev`
- 已安装 FastBEV 推理开发所需的核心依赖:
  `torch 1.10.1+cu111`、`torchvision 0.11.2+cu111`、`torchaudio 0.10.1`、
  `mmcv-full 1.6.0`、`mmdet 2.25.1`、`mmsegmentation 0.25.0`
- 已支持仓库 editable install
- 已增加一个单样本推理脚本:
  [tools/fastbev_infer.py](/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/tools/fastbev_infer.py)

当前机器的特别说明:

- 本机 CUDA toolkit 是 `12.3`
- 当前 PyTorch 是 `cu111`
- 因此本机不能直接编译 `bev_pool_v2` 的原生 CUDA 扩展
- 仓库已经加入纯 PyTorch fallback，可继续做功能开发、流程验证和结果导出
- 但这一路径推理速度会慢于原生 CUDA 扩展
- 如果命令里指定 `--device cuda:0` 但当前环境无可用 GPU，脚本会自动回退到 CPU

详细说明见 [LOCAL_SETUP.md](/home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev/LOCAL_SETUP.md)

## 2. 环境激活

项目使用的是项目内 conda envs 目录，不污染全局环境。

```bash
cd /home/ego_vehicle/MY_project/BEV_perception/advanced-fastbev-fastbev
source /home/ego_vehicle/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=$PWD/.conda_envs
export MPLCONFIGDIR=$PWD/.cache/matplotlib
conda activate bev
```

也可以直接用快捷脚本:

```bash
source /home/ego_vehicle/miniconda3/etc/profile.d/conda.sh
source ./activate_bev.sh
```

## 3. 推荐推理入口

推荐把下载好的模型统一放在仓库根目录下的 `model/` 目录，例如:

```bash
model/fastbev-r50-cbgs.pth
model/fastbev-r50-cbgs-4d.pth
```

如果目录还没有，可以先创建:

```bash
mkdir -p model
```

后续开发优先使用这个脚本:

```bash
python tools/fastbev_infer.py \
  configs/fastbev/paper/fastbev-r50-cbgs.py \
  /path/to/checkpoint.pth \
  --sample-index 0 \
  --device cuda:0 \
  --score-thr 0.3
```

这个脚本会:

- 按 `config` 自动构建测试数据集和测试 pipeline
- 从 `cfg.data.test.ann_file` 中取指定样本
- 运行 FastBEV 单样本推理
- 输出结构化 JSON，便于后续做感知信息提取

默认输出目录:

```bash
outputs/fastbev_infer/
```

## 4. 推理脚本用法

### 4.1 按索引推理

```bash
python tools/fastbev_infer.py 
  configs/fastbev/paper/fastbev-r50-cbgs.py 
  model/fastbev-r50-cbgs.pth
  --sample-index 10 
  --device cuda:0
```

### 4.2 按 nuScenes sample token 推理

```bash
python tools/fastbev_infer.py 
  configs/fastbev/paper/fastbev-r50-cbgs.py 
  work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth 
  --sample-token <sample_token> 
  --device cuda:0
```

### 4.3 指定输出路径

```bash
python tools/fastbev_infer.py \
  configs/fastbev/paper/fastbev-r50-cbgs.py \
  work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth \
  --sample-index 0 \
  --out outputs/fastbev_infer/sample0.json
```

### 4.4 常用参数

- `--sample-index`: 从测试标注文件中按序号取样本
- `--sample-token`: 按 nuScenes token 取样本
- `--device`: 例如 `cuda:0` 或 `cpu`
- `--score-thr`: 导出框的分数阈值
- `--topk`: 最多导出多少个检测结果
- `--out-dir`: 默认 JSON 输出目录
- `--out`: 显式指定 JSON 文件路径
- `--cfg-options`: 临时覆盖配置项

### 4.5 当前已验证可运行

下面这条命令已经在当前工程里跑通，并成功导出 JSON:

```bash
python tools/fastbev_infer.py \
  configs/fastbev/paper/fastbev-r50-cbgs.py \
  model/fastbev-r50-cbgs.pth \
  --sample-index 10 \
  --device cuda:0
```

输出示例:

```bash
outputs/fastbev_infer/b6b0d9f2f2e14a3aaa2c8aedeb1edb69.json
```

## 5. JSON 输出格式

输出 JSON 主要包含:

- `sample.sample_token`
- `sample.camera_files`
- `num_detections`
- `detections`

每个检测结果包含:

- `label`
- `label_name`
- `score`
- `box_3d`

其中 `box_3d` 直接对应模型输出的 3D 框张量，适合后续继续做:

- 感知结果过滤
- 类别统计
- BEV 后处理
- 接口转发
- 自定义可视化

## 6. 训练与测试

### 单机训练

```bash
python tools/train.py configs/fastbev/paper/fastbev-r50-cbgs.py
```

### 多卡训练

```bash
./tools/dist_train.sh configs/fastbev/paper/fastbev-r50-cbgs.py 8
```

### 标准测试

```bash
python tools/test.py \
  configs/fastbev/paper/fastbev-r50-cbgs.py \
  /path/to/checkpoint.pth \
  --eval mAP
```

## 7. TensorRT 相关

仓库里已经保留了导出和测速脚本:

- `tools/convert_fastbev_to_TRT.py`
- `tools/analysis_tools/benchmark_trt_fastbev.py`

但在当前机器上，建议先把 PyTorch 推理链路和结果导出链路稳定下来，再继续做 TensorRT 部署。

## 8. 后续开发建议

如果你后面主要做“推理和感知信息获取”，建议下一步优先往这几个方向推进:

1. 扩展 `tools/fastbev_infer.py`，支持批量样本推理
2. 增加结果字段拆分，例如中心点、尺寸、朝向、速度
3. 增加按类别过滤、按分数过滤、按距离过滤
4. 增加 BEV 可视化和多相机叠加可视化
5. 增加面向下游模块的标准输出接口

## 9. 分阶段推进计划

为了把工程逐步推进到“感知实时推理”，建议按下面 4 个阶段走:

### 阶段 1：单样本推理打通

目标:

- 配置、模型、样本读取全部跑通
- 输出稳定的结构化 JSON
- 修完 checkpoint、CPU fallback、旧依赖兼容问题

当前状态:

- 已完成

### 阶段 2：结果工程化

目标:

- 把 `box_3d` 拆成更直接可用的字段
- 增加类别过滤、分数过滤、距离过滤
- 增加批量推理与结果汇总

推荐下一步:

- 增加 `--class-names`
- 增加 `--max-distance`
- 增加批量 token / index 列表输入

### 阶段 3：在线推理接口

目标:

- 把推理脚本改成长期运行的服务
- 输入一帧多相机数据，输出检测结果 JSON
- 为下游模块提供稳定接口

推荐形式:

- Python service
- FastAPI / Flask
- 单进程单模型常驻内存

### 阶段 4：实时优化

目标:

- 恢复 GPU 推理
- 评估 PyTorch 实时性能
- 继续推进 TensorRT / ONNX / 缓存优化

当前主要阻塞:

- 本机当前 PyTorch 无法访问可用 NVIDIA driver
- 当前 `bev_pool_v2` 使用的是纯 PyTorch fallback，不是原生 CUDA 扩展
- 所以当前适合功能开发，不适合真实实时性能评估

## 10. 模型与结果参考

![](./resources/fastbev++_exps.jpg)
![](./resources/fastbev++_result.png)

模型下载:

- [fastbev-r50-cbgs](https://drive.google.com/drive/folders/1AYtoX8XaNg8ZckFBgbo2TVmyXb26Dw2V?usp=sharing)
- [fastbev-r50-cbgs-4d](https://drive.google.com/drive/folders/1XBcftoEn_2TkpG-qQAo66Oa8_hEnfoqi?usp=sharing)
- [fastbev-r101-cbgs-4d-longterm](https://drive.google.com/drive/folders/1JLcU96Oimk7wSZLi7-FeImUGD_1fXWG1)
- [fastbev-r101-cbgs-4d-longterm-depth](https://drive.google.com/drive/folders/1RG2hFFu0germP-8sIvMYpv-K-eVER6yB)

## 11. 致谢

本项目建立在以下开源工作之上:

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVerse](https://github.com/zhangyp15/BEVerse)
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [Fast-BEV](https://github.com/Sense-GVT/Fast-BEV)
