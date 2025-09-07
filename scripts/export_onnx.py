#!/usr/bin/env python3
"""
Minimal wrapper to export YOLOv5 .pt -> ONNX using Ultralytics' exporter.

Requirements:
- A local YOLOv5 checkout containing export.py (recommended): https://github.com/ultralytics/yolov5
  e.g., clone next to this repo and install its requirements.

Usage:
  python scripts/export_onnx.py \
    --weights path/to/weights.pt --imgsz 640 \
    --yolov5-dir ./yolov5 --out models/yolov5s.onnx --opset 12

Notes:
- This wrapper avoids re-implementing Ultralytics' export logic and ensures a correct ONNX graph.
- If you prefer, you can call yolov5/export.py directly.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to YOLOv5 .pt weights")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--yolov5-dir", required=True, help="Path to local YOLOv5 repo (must contain export.py)")
    ap.add_argument("--out", default="models/model.onnx", help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    ap.add_argument("--dynamic", action="store_true", help="Export with dynamic batch axis (if supported)")
    args = ap.parse_args()

    ydir = Path(args.yolov5_dir)
    export_py = ydir / "export.py"
    if not export_py.exists():
        sys.exit(f"export.py not found at: {export_py}. Clone Ultralytics YOLOv5 and point --yolov5-dir there.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run Ultralytics exporter
    cmd = [
        sys.executable,
        str(export_py),
        "--weights", str(args.weights),
        "--imgsz", str(args.imgsz),
        "--include", "onnx",
        "--opset", str(args.opset),
    ]
    if args.dynamic:
        cmd += ["--dynamic"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ydir))

    # Find the ONNX that was just produced in yolov5 dir
    onnx_candidates = sorted(ydir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not onnx_candidates:
        sys.exit("No .onnx produced by exporter. Check exporter output above.")
    latest = onnx_candidates[0]
    print(f"Copying {latest} -> {out_path}")
    shutil.copy2(latest, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
