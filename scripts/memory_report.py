#!/usr/bin/env python3
"""
Memory footprint report for ONNX Runtime (and optionally OpenVINO) inference.

Reports process RSS before and after model load and after a warmup inference.

Example:
  python scripts/memory_report.py --onnx models/yolov5s.onnx --imgsz 640 --intra 8 --inter 4
"""
import argparse
import os

import numpy as np


def rss_mb():
    try:
        import psutil  # type: ignore
    except Exception as e:
        raise SystemExit("psutil is required. pip install psutil") from e
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def onnx_report(path, imgsz, intra, inter):
    import onnxruntime as ort
    m0 = rss_mb()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra
    sess_options.inter_op_num_threads = inter
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_type = sess.get_inputs()[0].type or "tensor(float)"
    expected_dtype = np.float16 if "float16" in input_type else np.float32
    m1 = rss_mb()
    x = np.random.rand(1, 3, imgsz, imgsz).astype(expected_dtype)
    for _ in range(3):
        _ = sess.run([sess.get_outputs()[0].name], {input_name: x})
    m2 = rss_mb()
    return m0, m1, m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--intra", type=int, default=8)
    ap.add_argument("--inter", type=int, default=4)
    args = ap.parse_args()

    m0, m1, m2 = onnx_report(args.onnx, args.imgsz, args.intra, args.inter)
    print("ONNX Runtime RSS (MB)")
    print(f"Before load: {m0:.1f}")
    print(f"After load:  {m1:.1f}")
    print(f"After warmup:{m2:.1f}")


if __name__ == "__main__":
    main()

