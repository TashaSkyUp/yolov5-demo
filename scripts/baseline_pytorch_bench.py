#!/usr/bin/env python3
"""
Baseline PyTorch CPU benchmark for YOLOv5s using torch.hub.

This script measures model-only forward latency with random input on CPU.
Requires: torch (CPU) and network access for torch.hub (first run).

Example:
  python scripts/baseline_pytorch_bench.py --imgsz 640 --runs 30 --warmup 5
"""
import argparse
import statistics as stats
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import torch

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = model.autoshape(False)  # raw forward
    model.eval()
    model.to('cpu')

    x = torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float32, device='cpu')

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)

    times = []
    with torch.inference_mode():
        for _ in range(args.runs):
            t0 = time.time()
            _ = model(x)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)

    mean_ms = stats.fmean(times)
    p50 = float(np.percentile(times, 50))
    p90 = float(np.percentile(times, 90))
    print(f"Runs: {args.runs} Warmup: {args.warmup}")
    print(f"Latency ms -> mean: {mean_ms:.2f}, p50: {p50:.2f}, p90: {p90:.2f}, min: {min(times):.2f}, max: {max(times):.2f}")
    if mean_ms > 0:
        print(f"Throughput (1-batch): {1000.0/mean_ms:.2f} FPS")


if __name__ == "__main__":
    main()

