#!/usr/bin/env python3
"""
Client-facing step-by-step demo script.

What it does (with friendly, visible steps):
  1) Ensure/download YOLOv5s ONNX model
  2) Ensure a 15-class list (random from COCO) and show it
  3) Generate a sample image
  4) Benchmark ONNX Runtime (model-only)
  5) Dynamic batching demo (notes if model has static batch=1)
  6) Memory footprint report (ORT)
  7) INT8 calibration (OpenVINO, with correct normalization)
  8) OpenVINO FP32 vs INT8 benchmark and speedup
  9) Run an example inference restricted to 15 classes and save a visualization

Usage (typical):
  python scripts/client_demo.py \
    --onnx models/yolov5s.onnx --imgsz 640 --runs 50 --warmup 10 --intra 8 --inter 4

All steps print banners and concise results suitable for live demos.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print("\n====> $", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if check and proc.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return proc


def banner(title):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def ensure_onnx(onnx_path: Path):
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"ONNX model found: {onnx_path}")
        return
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
        "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx",
        "https://huggingface.co/ultralytics/yolov5/resolve/main/yolov5s.onnx",
    ]
    for u in urls:
        try:
            banner(f"Downloading ONNX model from {u}")
            run(["curl", "-L", "-o", str(onnx_path), u])
            if onnx_path.exists() and onnx_path.stat().st_size > 0:
                print(f"Downloaded ONNX to {onnx_path}")
                return
        except SystemExit:
            continue
    raise SystemExit("Failed to download ONNX model from known sources.")


def ensure_15_classes():
    coco = Path("labels/coco80.txt")
    if not coco.exists():
        banner("Creating COCO 80 label file")
        coco.parent.mkdir(parents=True, exist_ok=True)
        coco.write_text("\n".join([
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
            "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        ]) + "\n")
    classes15 = Path("labels/classes_15.txt")
    if not classes15.exists():
        banner("Selecting 15 random classes from COCO")
        run([sys.executable, "scripts/generate_15_classes.py"])  # prints selection
    else:
        print("15-class list present:")
        print(classes15.read_text())
    return classes15


def ensure_sample_image():
    sample = Path("samples/sample.jpg")
    if sample.exists():
        print(f"Sample image present: {sample}")
        return sample
    banner("Generating synthetic sample image")
    run([sys.executable, "scripts/gen_sample_image.py"])
    return sample


def ensure_calib_images(n=200, imgsz=640):
    d = Path("calib_images")
    if d.exists() and any(d.glob("*.jpg")):
        print(f"Calibration directory present: {d}")
        return d
    banner(f"Generating synthetic calibration set: {n} images of size {imgsz}")
    import cv2
    import numpy as np
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
        cv2.putText(img, f"calib {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(d / f"img_{i:04d}.jpg"), img)
    print(f"Wrote {n} images under {d}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/yolov5s.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--intra", type=int, default=8)
    ap.add_argument("--inter", type=int, default=4)
    ap.add_argument("--int8-dir", default="models/int8")
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    int8_dir = Path(args.int8_dir)

    banner("Step 1: Ensure/download YOLOv5s ONNX model")
    ensure_onnx(onnx_path)

    banner("Step 2: Ensure 15-class list and show it")
    classes15 = ensure_15_classes()

    banner("Step 3: Generate sample image")
    sample = ensure_sample_image()

    banner("Step 4: ONNX Runtime quick model-only benchmark")
    run([
        sys.executable, "scripts/quick_bench.py",
        "--onnx", str(onnx_path), "--imgsz", str(args.imgsz),
        "--runs", str(args.runs), "--warmup", str(args.warmup),
        "--intra", str(args.intra), "--inter", str(args.inter),
        "--random",
    ])

    banner("Step 5: Dynamic batching demo (ORT)")
    run([
        sys.executable, "scripts/dynamic_batcher_demo.py",
        "--onnx", str(onnx_path), "--imgsz", str(args.imgsz),
        "--batches", "1", "2", "4", "8",
        "--runs", "50", "--warmup", "10",
        "--intra", str(args.intra), "--inter", str(args.inter),
    ])

    banner("Step 6: Memory footprint report (ORT)")
    run([
        sys.executable, "scripts/memory_report.py",
        "--onnx", str(onnx_path), "--imgsz", str(args.imgsz),
        "--intra", str(args.intra), "--inter", str(args.inter),
    ])

    if not args.skip_quant:
        banner("Step 7: INT8 calibration (OpenVINO, with normalization fix)")
        calib_dir = ensure_calib_images(n=200, imgsz=args.imgsz)
        int8_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "scripts/calibrate_openvino_int8.py",
            "--model", str(onnx_path),
            "--data-dir", str(calib_dir),
            "--output-dir", str(int8_dir),
            "--subset", "200",
            "--preset", "performance",
            "--imgsz", str(args.imgsz),
            "--normalize",
            "--mean", "0.485", "0.456", "0.406",
            "--std", "0.229", "0.224", "0.225",
        ])

        banner("Step 8: OpenVINO FP32 vs INT8 benchmark")
        int8_xml = int8_dir / "model_int8.xml"
        run([
            sys.executable, "scripts/openvino_bench.py",
            "--model", str(onnx_path),
            "--int8-model", str(int8_xml),
            "--imgsz", str(args.imgsz),
            "--runs", str(args.runs), "--warmup", str(args.warmup),
        ])

    banner("Step 9: Example inference restricted to 15 classes (save visualization)")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "scripts/infer_onnx.py",
        "--onnx", str(onnx_path),
        "--image", str(sample),
        "--imgsz", str(args.imgsz),
        "--conf", "0.25", "--iou", "0.45",
        "--intra", str(args.intra), "--inter", str(args.inter),
        "--save-vis", str(out_dir),
        "--allowed-classes-file", "labels/classes_15.txt",
    ])

    banner("Demo complete")
    print("- ONNX model:", onnx_path)
    print("- 15 classes file: labels/classes_15.txt (may vary per run)")
    print("- Sample image:", sample)
    print("- Visualization saved under:", out_dir)
    if not args.skip_quant:
        print("- INT8 IR saved under:", int8_dir)


if __name__ == "__main__":
    main()

