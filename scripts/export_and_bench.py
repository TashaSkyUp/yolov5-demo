#!/usr/bin/env python3
"""
Helper to export a YOLOv5 model to ONNX (optional) and run a quick latency benchmark.

Two modes:
1) Export then bench:
   python scripts/export_and_bench.py \
     --weights path/to/weights.pt --yolov5-dir ./yolov5 \
     --imgsz 640 --opset 12 --runs 50 --warmup 10 --intra 8 --inter 4

2) Direct bench from existing ONNX:
   python scripts/export_and_bench.py \
     --onnx models/yolov5s.onnx --runs 50 --warmup 10 --intra 8 --inter 4 --random

Notes:
- Uses scripts/export_onnx.py and scripts/quick_bench.py under the hood.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", help="Path to YOLOv5 .pt weights (triggers export)")
    g.add_argument("--onnx", help="Path to existing ONNX model (skip export)")

    ap.add_argument("--yolov5-dir", help="Path to YOLOv5 repo (required with --weights)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default="models/yolov5s.onnx", help="Output ONNX path for export")

    # Benchmark args
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--intra", type=int, default=8)
    ap.add_argument("--inter", type=int, default=4)
    ap.add_argument("--img", help="Benchmark image path; if omitted, use --random or samples/sample.jpg if present")
    ap.add_argument("--random", action="store_true", help="Use random input for benchmarking")
    args = ap.parse_args()

    onnx_path = args.onnx
    if args.weights:
        if not args.yolov5_dir:
            sys.exit("--yolov5-dir is required when using --weights for export")
        onnx_path = args.out
        run([
            sys.executable, "scripts/export_onnx.py",
            "--weights", args.weights,
            "--imgsz", str(args.imgsz),
            "--yolov5-dir", args.yolov5_dir,
            "--out", onnx_path,
            "--opset", str(args.opset),
        ])

    # Benchmark
    bench_cmd = [
        sys.executable, "scripts/quick_bench.py",
        "--onnx", onnx_path,
        "--imgsz", str(args.imgsz),
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--intra", str(args.intra),
        "--inter", str(args.inter),
    ]
    if args.random:
        bench_cmd.append("--random")
    elif args.img:
        bench_cmd.extend(["--img", args.img])
    run(bench_cmd)


if __name__ == "__main__":
    main()

