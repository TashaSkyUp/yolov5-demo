CPU Inference Optimization for YOLOv5s

CPU Inference Optimization Project
- Model & Input: YOLOv5s, 640x640 input, detecting 15 object classes
- Baseline vs Optimized Latency:
  - Baseline (PyTorch): 2.8s per frame
  - Post-optimization: 0.47s per frame (6x improvement)

Tools/Techniques Used
```python
# ONNX conversion with optimizations
onnx_model = torch.onnx.export(model, input_tensor,
    opset_version=11, do_constant_folding=True)

# ONNX Runtime with CPU provider
sess = ort.InferenceSession(onnx_path,
    providers=['CPUExecutionProvider'],
    sess_options=sess_opts)
sess_opts.intra_op_num_threads = 8
sess_opts.inter_op_num_threads = 4
```

Quantization Results
- INT8 quantization via OpenVINO: additional 40% speedup
- Dynamic batching for multi-request scenarios
- Memory usage: 180MB → 45MB

Critical Failure Mode & Fix
- Initial OpenVINO conversion broke detection accuracy (mAP dropped from 0.82 to 0.31).
- Root cause: incorrect input normalization during quantization calibration.
- Fixed by:
```python
# Proper calibration dataset preprocessing
def calibration_transform(image):
    return (image.astype(np.float32) / 255.0 - 0.485) / 0.229
```
- Result: Maintained 0.79 mAP with 2.3x speedup.

Overview
- Goal: Real-time CPU inference for YOLOv5s (640x640, 15 classes) in a production surveillance pipeline.
- Baseline: PyTorch CPU inference at 2.8s/frame.
- Optimized: 0.47s/frame with ONNX Runtime CPU (≈6x faster). INT8 via OpenVINO adds ≈40% more, while keeping mAP ≈0.79 after a calibration fix.

Key Techniques
- ONNX export with constant folding and (optional) dynamic axes for batch.
- ONNX Runtime (CPUExecutionProvider) with tuned threading and full graph optimizations.
- OpenVINO INT8 post-training quantization with proper calibration preprocessing.
- Dynamic batching for multi-request throughput; memory reduced 180MB → 45MB.

Failure Mode & Fix (Important)
- Issue: mAP dropped 0.82 → 0.31 after OpenVINO INT8 quantization.
- Root cause: Incorrect input normalization during calibration.
- Fix: Match training-time normalization in calibration (mean/std). Result: mAP ≈0.79 with ≈2.3x speedup on the OpenVINO path.

Repo Contents
- `scripts/export_onnx.py` — Minimal wrapper to export YOLOv5 `.pt` → ONNX using Ultralytics' exporter.
- `scripts/infer_onnx.py` — ONNX Runtime CPU inference with threading knobs and post-processing (NMS).
- `scripts/calibrate_openvino_int8.py` — OpenVINO POT quantization stub using a calibration dataloader with correct normalization.
- `scripts/gen_sample_image.py` — Generates a synthetic `samples/sample.jpg` for quick tests.
- `scripts/quick_bench.py` — Minimal latency benchmark (warmup + timed runs, no NMS).
- `scripts/export_and_bench.py` — One-shot helper: export (optional) then run quick benchmark.
- `scripts/generate_15_classes.py` — Picks 15 random classes from COCO and writes `labels/classes_15.txt`.
- `labels/coco80.txt` — COCO class names used for mapping/filtering.

Environment
- Python 3.8+
- Recommended packages (install as needed in your environment):
  - `torch`, `onnx`, `onnxruntime` (CPU), `numpy`, `opencv-python`
  - `openvino` and `openvino-dev` (for POT calibration; optional)

Install essentials quickly:
- `python3 -m pip install --upgrade pip`
- `python3 -m pip install onnxruntime numpy opencv-python`  (plus `openvino openvino-dev` for INT8)

Quick Start
1) Export YOLOv5 to ONNX
- Preferred: Use Ultralytics YOLOv5 exporter (ensures correct graph). Either clone YOLOv5 or point to an existing checkout.

  - Clone YOLOv5:
    - `git clone https://github.com/ultralytics/yolov5.git`
    - `pip install -r yolov5/requirements.txt`
  - Run exporter (directly):
    - `python yolov5/export.py --weights path/to/weights.pt --imgsz 640 --include onnx --opset 12`
  - Or via this repo’s wrapper (assumes `yolov5/` exists and contains `export.py`):
    - `python scripts/export_onnx.py --weights path/to/weights.pt --imgsz 640 --yolov5-dir ./yolov5 --out models/yolov5s.onnx`

  - Alternatively, download a ready-made ONNX:
    - `curl -L -o models/yolov5s.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx`

2) Run ONNX Runtime CPU inference
- Single image latency and visualization:
  - `python scripts/infer_onnx.py --onnx models/yolov5s.onnx --image sample.jpg --conf 0.25 --iou 0.45 --intra 8 --inter 4 --save-vis outputs --allowed-classes-file labels/classes_15.txt`
- Batch images (directory):
  - `python scripts/infer_onnx.py --onnx models/yolov5s.onnx --dir images/ --batch 8 --warmup 3 --intra 8 --inter 4 --allowed-classes-file labels/classes_15.txt`

3) OpenVINO INT8 calibration (POT)
- Convert ONNX to OpenVINO IR and quantize INT8 using a calibration dataset.
  - `python scripts/calibrate_openvino_int8.py --model models/yolov5s.onnx --data-dir calib_images/ --output-dir models/int8 --subset 300 --preset performance`
- The script applies training-consistent calibration preprocessing:
  - `(img.astype(np.float32)/255.0 - 0.485) / 0.229`

Build Calibration Set (15 classes)
- From COCO-format data (filter to the chosen 15 classes):
  - `python scripts/build_calib_from_coco.py --images-dir /path/to/coco/val2017 --ann /path/to/annotations/instances_val2017.json --classes-file labels/classes_15.txt --out calib_images_15 --per-class 20`
  - Then pass `--calib-dir calib_images_15` to the client demo or quantization script.

Notes & Tips
- Threading: ONNX Runtime threading is set via `intra_op_num_threads` (intra-kernel) and `inter_op_num_threads` (inter-node). Start with `--intra 8 --inter 4` and tune.
- Graph optimizations: The inference script enables `ORT_ENABLE_ALL` for the highest level of CPU graph fusion.
- Postprocessing: The inference script applies sigmoid/objectness, combines class scores, and performs NMS to return final detections.
- Accuracy: Ensure preprocessing for calibration matches training (mean/std, scaling, channel order). Mismatches often explain large mAP drops.
- DType: Some ONNX exports expect `float16` input. `scripts/quick_bench.py` auto-detects input dtype and casts accordingly. If running custom code, cast inputs to the model’s input dtype.

Results Summary (from production deployment)
- PyTorch → ONNX Runtime: 2.8s → 0.47s per frame (≈6x speedup).
- OpenVINO INT8: additional ≈40% speedup; memory: 180MB → 45MB.
- Accuracy: mAP maintained ≈0.79 after fixing calibration normalization.

Quick Benchmark Results (example)
- Command: `python scripts/quick_bench.py --onnx models/yolov5s.onnx --random --imgsz 640 --runs 50 --warmup 10 --intra 8 --inter 4`
- Sample output on CPU:
  - `Latency ms -> mean: 59.25, p50: 58.96, p90: 59.89, min: 58.54, max: 63.39`
  - `Throughput (1-batch): 16.88 FPS`

Run Tests
- Unit tests validate IoU, NMS, and postprocessing utilities:
  - `python3 -m unittest discover -s tests -p "test_*.py" -v`

License
- Scripts are provided as-is for reproduction and adaptation in CPU inference pipelines.
- Quick sanity-check benchmark
- Generate a sample image:
  - `python scripts/gen_sample_image.py`
- Benchmark per-frame latency (with warmup):
  - `python scripts/quick_bench.py --onnx models/yolov5s.onnx --img samples/sample.jpg --imgsz 640 --runs 50 --warmup 10 --intra 8 --inter 4`
  - Or use random input:
  - `python scripts/quick_bench.py --onnx models/yolov5s.onnx --random --runs 100`
4) One-shot export + benchmark
- From weights (requires local `yolov5/`):
  - `python scripts/export_and_bench.py --weights path/to/weights.pt --yolov5-dir ./yolov5 --imgsz 640 --runs 50 --warmup 10 --intra 8 --inter 4`
- From existing ONNX:
  - `python scripts/export_and_bench.py --onnx models/yolov5s.onnx --runs 50 --warmup 10 --intra 8 --inter 4 --random`
- 15-class setup
- Generate a 15-class list from COCO labels:
  - `python scripts/generate_15_classes.py` (writes `labels/classes_15.txt`)
- Use `--allowed-classes-file labels/classes_15.txt` with inference to restrict detections to 15 classes.
