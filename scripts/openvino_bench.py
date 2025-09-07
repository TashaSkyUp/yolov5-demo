#!/usr/bin/env python3
"""
OpenVINO runtime latency benchmark for FP32 (from ONNX) and INT8 IR.

Examples:
  # FP32 from ONNX
  python scripts/openvino_bench.py --model models/yolov5s.onnx --imgsz 640 --runs 50 --warmup 10

  # Compare FP32 vs INT8
  python scripts/openvino_bench.py --model models/yolov5s.onnx --int8-model models/int8/model_int8.xml --imgsz 640 --runs 50 --warmup 10
"""
import argparse
import statistics as stats
import time

import numpy as np


def bench_model(core, model_path, imgsz, runs, warmup):
    model = core.read_model(model_path)
    compiled = core.compile_model(model, device_name="CPU")
    inp = compiled.inputs[0]
    et = inp.get_element_type().get_type_name() if hasattr(inp.get_element_type(), 'get_type_name') else str(inp.get_element_type())
    dtype = np.float16 if 'f16' in et else np.float32
    shape = list(inp.get_partial_shape().get_shape()) if hasattr(inp, 'get_partial_shape') else list(inp.shape)
    if shape[0] in (None, -1):
        shape[0] = 1
    x = np.random.rand(*shape).astype(dtype)
    req = compiled.create_infer_request()
    # Warmup
    for _ in range(warmup):
        req.infer({inp: x})
    times = []
    for _ in range(runs):
        t0 = time.time()
        req.infer({inp: x})
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return {
        'mean': stats.fmean(times),
        'p50': float(np.percentile(times, 50)),
        'p90': float(np.percentile(times, 90)),
        'min': min(times),
        'max': max(times),
        'fps': 1000.0 / stats.fmean(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='FP32 model path (ONNX or IR .xml)')
    ap.add_argument('--int8-model', help='INT8 IR path (.xml) for comparison')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    args = ap.parse_args()

    from openvino.runtime import Core
    core = Core()

    print('FP32/FP model:', args.model)
    fp = bench_model(core, args.model, args.imgsz, args.runs, args.warmup)
    print(f"FP mean {fp['mean']:.2f} ms | p50 {fp['p50']:.2f} | p90 {fp['p90']:.2f} | FPS {fp['fps']:.2f}")

    if args.int8_model:
        print('INT8 model:', args.int8_model)
        int8 = bench_model(core, args.int8_model, args.imgsz, args.runs, args.warmup)
        print(f"INT8 mean {int8['mean']:.2f} ms | p50 {int8['p50']:.2f} | p90 {int8['p90']:.2f} | FPS {int8['fps']:.2f}")
        print(f"Speedup: {fp['mean']/int8['mean']:.2f}x")


if __name__ == '__main__':
    main()

