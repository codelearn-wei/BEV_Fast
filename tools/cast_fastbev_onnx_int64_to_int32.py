#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cast safe INT64 constants and selected inputs in FastBEV ONNX to INT32')
    parser.add_argument('src', help='source onnx path')
    parser.add_argument('dst', help='destination onnx path')
    parser.add_argument(
        '--input-names',
        nargs='*',
        default=['coors_img', 'coors_depth'],
        help='graph input names to cast from INT64 to INT32')
    return parser.parse_args()


def safe_to_int32(arr):
    if arr.dtype != np.int64:
        return None
    if arr.size == 0:
        return arr.astype(np.int32)
    if arr.min() < INT32_MIN or arr.max() > INT32_MAX:
        return None
    return arr.astype(np.int32)


def cast_initializer(initializer):
    arr = numpy_helper.to_array(initializer)
    arr32 = safe_to_int32(arr)
    if arr32 is None:
        return False
    initializer.CopyFrom(numpy_helper.from_array(arr32, name=initializer.name))
    return True


def cast_constant_node(node):
    changed = False
    for attr in node.attribute:
        if attr.name != 'value':
            continue
        arr = numpy_helper.to_array(attr.t)
        arr32 = safe_to_int32(arr)
        if arr32 is None:
            continue
        attr.t.CopyFrom(numpy_helper.from_array(arr32, name=attr.t.name))
        changed = True
    return changed


def cast_graph_inputs(graph, input_names):
    changed = 0
    for value_info in graph.input:
        if value_info.name not in input_names:
            continue
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == TensorProto.INT64:
            tensor_type.elem_type = TensorProto.INT32
            changed += 1
    return changed


def main():
    args = parse_args()
    model = onnx.load(args.src)

    init_changed = sum(cast_initializer(init) for init in model.graph.initializer)
    const_changed = sum(cast_constant_node(node) for node in model.graph.node)
    input_changed = cast_graph_inputs(model.graph, set(args.input_names))

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_path))
    print(f'initializers_cast: {init_changed}')
    print(f'constants_cast: {const_changed}')
    print(f'inputs_cast: {input_changed}')
    print(f'output_onnx: {dst_path}')


if __name__ == '__main__':
    main()
