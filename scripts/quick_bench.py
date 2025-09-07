#!/usr/bin/env python3
"""
Quick ONNX Runtime latency benchmark for YOLOv5 ONNX models.

Runs warmup + timed runs over a single image (or random input) and reports
per-frame latency statistics. No postprocessing (NMS) to isolate model time.

Usage:
  python scripts/quick_bench.py --onnx models/yolov5s.onnx --img samples/sample.jpg \
    --imgsz 640 --runs 50 --warmup 10 --intra 8 --inter 4
  python scripts/quick_bench.py --onnx models/yolov5s.onnx --random --runs 100
"""
import argparse
import statistics as stats
import time
from pathlib import Path

import numpy as np


def lazy_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception as e:
        raise SystemExit("OpenCV (cv2) is required for image input. Use --random to bypass.") from e


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    cv2 = lazy_import_cv2()
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def preprocess_image(path, imgsz=640, normalize=False, mean=None, std=None):
    cv2 = lazy_import_cv2()
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"Failed to read image: {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    lb = letterbox(im, new_shape=(imgsz, imgsz))
    img = lb.astype(np.float32) / 255.0
    if normalize:
        mean = np.array(mean or [0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std or [1.0, 1.0, 1.0], dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--img", help="Path to an input image")
    g.add_argument("--random", action="store_true", help="Use random input instead of an image")
    ap.add_argument("--imgsz", type=int, default=640, help="Model input size")
    ap.add_argument("--runs", type=int, default=50, help="# of timed runs")
    ap.add_argument("--warmup", type=int, default=10, help="# of warmup runs")
    ap.add_argument("--intra", type=int, default=8, help="ONNX Runtime intra-op threads")
    ap.add_argument("--inter", type=int, default=4, help="ONNX Runtime inter-op threads")
    ap.add_argument("--normalize", action="store_true", help="Apply mean/std normalization to image")
    ap.add_argument("--mean", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Mean (RGB)")
    ap.add_argument("--std", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Std (RGB)")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise SystemExit("onnxruntime is required. pip install onnxruntime") from e

    # Session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra
    sess_options.inter_op_num_threads = args.inter
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    # Determine expected dtype
    input_type = sess.get_inputs()[0].type or "tensor(float)"
    expected_dtype = np.float16 if "float16" in input_type else np.float32

    # Prepare input
    if args.random:
        x = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(expected_dtype)
    else:
        if not args.img:
            # default to sample if present
            sample = Path("samples/sample.jpg")
            if sample.exists():
                args.img = str(sample)
            else:
                raise SystemExit("Provide --img or use --random or run scripts/gen_sample_image.py")
        x = preprocess_image(args.img, imgsz=args.imgsz, normalize=args.normalize, mean=args.mean, std=args.std).astype(expected_dtype)

    # Warmup
    for _ in range(args.warmup):
        _ = sess.run(output_names, {input_name: x})

    # Timed
    times_ms = []
    for _ in range(args.runs):
        t0 = time.time()
        _ = sess.run(output_names, {input_name: x})
        t1 = time.time()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = stats.fmean(times_ms)
    p50 = np.percentile(times_ms, 50)
    p90 = np.percentile(times_ms, 90)
    print("Runs:", args.runs, "Warmup:", args.warmup)
    print(f"Latency ms -> mean: {mean_ms:.2f}, p50: {p50:.2f}, p90: {p90:.2f}, min: {min(times_ms):.2f}, max: {max(times_ms):.2f}")
    if mean_ms > 0:
        print(f"Throughput (1-batch): {1000.0/mean_ms:.2f} FPS")


if __name__ == "__main__":
    main()
