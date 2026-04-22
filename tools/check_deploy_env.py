#!/usr/bin/env python3
import importlib.util
import os
import platform
import subprocess
import sys


REQUIRED_PYTHON_PACKAGES = ['onnx', 'onnxsim']
OPTIONAL_PYTHON_PACKAGES = ['tensorrt', 'mmdeploy']


def has_module(name):
    return importlib.util.find_spec(name) is not None


def print_section(title):
    print(f'\n== {title} ==')


def safe_run(cmd):
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False)
        return result.returncode, result.stdout.strip()
    except Exception as exc:
        return 1, str(exc)


def main():
    print_section('Python')
    print(f'python: {sys.version.split()[0]}')
    print(f'platform: {platform.platform()}')

    print_section('Packages')
    for name in REQUIRED_PYTHON_PACKAGES:
        print(f'{name}: {"ok" if has_module(name) else "missing"}')
    for name in OPTIONAL_PYTHON_PACKAGES:
        print(f'{name}: {"ok" if has_module(name) else "missing"}')

    print_section('CUDA / WSL')
    print(f'CUDA_HOME: {os.environ.get("CUDA_HOME", "(unset)")}')
    code, output = safe_run(['/usr/lib/wsl/lib/nvidia-smi'])
    if code == 0:
        print('wsl nvidia-smi: ok')
        print(output)
    else:
        code, output = safe_run(['nvidia-smi'])
        print(f'nvidia-smi: {"ok" if code == 0 else "missing"}')
        if output:
            print(output)

    print_section('Next Steps')
    print('1. 安装 ONNX 基础依赖')
    print('   pip install onnx==1.13.1 onnxsim==0.4.36')
    print('2. 确认 TensorRT Python wheel 与系统 TensorRT 版本一致')
    print('3. 如需 benchmark_trt_fastbev.py，还需要 mmdeploy')
    print('4. 导出 ONNX 后，再生成 TensorRT engine 做测速对比')


if __name__ == '__main__':
    main()
