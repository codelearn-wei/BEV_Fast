#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Replace INT64_MAX slice-end constants in exported FastBEV ONNX')
    parser.add_argument('src', help='source onnx path')
    parser.add_argument('dst', help='destination onnx path')
    parser.add_argument(
        '--replacement',
        type=int,
        default=2147483647,
        help='replacement value for INT64_MAX slice-end constants')
    return parser.parse_args()


def main():
    args = parse_args()
    model = onnx.load(args.src)
    replaced = 0
    target = np.iinfo(np.int64).max

    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        for attr in node.attribute:
            if attr.name != 'value':
                continue
            arr = numpy_helper.to_array(attr.t)
            if arr.dtype == np.int64 and arr.shape == (1,) and int(arr[0]) == target:
                new_arr = np.array([args.replacement], dtype=np.int64)
                attr.t.CopyFrom(numpy_helper.from_array(new_arr, name=attr.t.name))
                replaced += 1

    if replaced == 0:
        raise RuntimeError('No INT64_MAX slice-end constants were found.')

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_path))
    print(f'replaced_constants: {replaced}')
    print(f'output_onnx: {dst_path}')


if __name__ == '__main__':
    main()
