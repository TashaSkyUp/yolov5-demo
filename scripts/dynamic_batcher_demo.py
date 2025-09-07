#!/usr/bin/env python3
"""
Dynamic batching throughput demo using ONNX Runtime CPU.

Runs random inputs at various batch sizes and reports throughput (FPS) and per-image latency.

Example:
  python scripts/dynamic_batcher_demo.py --onnx models/yolov5s.onnx --imgsz 640 \
    --batches 1 2 4 8 --runs 100 --warmup 10 --intra 8 --inter 4
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--intra", type=int, default=8)
    ap.add_argument("--inter", type=int, default=4)
    args = ap.parse_args()

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra
    sess_options.inter_op_num_threads = args.inter
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_type = inp.type or "tensor(float)"
    expected_dtype = np.float16 if "float16" in input_type else np.float32
    output_names = [o.name for o in sess.get_outputs()]
    # Detect if batch is dynamic; if not, force to 1
    shape = inp.shape
    dynamic_batch = isinstance(shape[0], str) or shape[0] is None
    if not dynamic_batch and shape[0] != 1:
        # rare case: fixed batch >1; respect it
        fixed_bs = int(shape[0])
        print(f"Note: model has fixed batch={fixed_bs}. Using that.")
        args.batches = [fixed_bs]
    elif not dynamic_batch and shape[0] == 1:
        print("Note: model has fixed batch=1. Using batch=1 only. Re-export with dynamic axes for multi-batch.")
        args.batches = [1]

    results = []
    for bs in args.batches:
        x = np.random.rand(bs, 3, args.imgsz, args.imgsz).astype(expected_dtype)
        # warmup
        for _ in range(args.warmup):
            _ = sess.run(output_names, {input_name: x})
        # timed
        t0 = time.time()
        for _ in range(args.runs):
            _ = sess.run(output_names, {input_name: x})
        t1 = time.time()
        total_ms = (t1 - t0) * 1000.0
        per_img_ms = total_ms / (args.runs * bs)
        fps = 1000.0 / per_img_ms
        results.append((bs, per_img_ms, fps))

    print("BatchSize, PerImage(ms), Throughput(FPS)")
    for bs, ms, fps in results:
        print(f"{bs}, {ms:.2f}, {fps:.2f}")


if __name__ == "__main__":
    main()
