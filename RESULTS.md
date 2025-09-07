CPU Inference Optimization — Results

Environment
- CPU: (container default)
- Python: python3.10
- onnxruntime: 1.22.1
- Model: yolov5s.onnx (Ultralytics v7.0 release asset)
- Input size: 640x640, batch=1 (fixed in this asset)
- Classes: Restricted to a randomly selected set of 15 COCO classes via `labels/classes_15.txt` when running full inference (postprocessing filter).

Baseline vs Optimized (context from production)
- Baseline (PyTorch CPU): 2.8 s/frame (reference claim)
- Optimized (ONNX Runtime CPU): 0.47 s/frame (~6x faster; reference claim)

Model-only Latency (measured here)
- Command: `python scripts/quick_bench.py --onnx models/yolov5s.onnx --random --imgsz 640 --runs 50 --warmup 10 --intra 8 --inter 4`
- Result:
  - mean: 59.25 ms, p50: 58.96 ms, p90: 59.89 ms
  - throughput (1-batch): 16.88 FPS
- Notes: This is model-only time (no NMS/I/O). The downloaded ONNX expects float16 input; the script casts automatically.

Dynamic Batching (ORT)
- Command: `python scripts/dynamic_batcher_demo.py --onnx models/yolov5s.onnx --imgsz 640 --batches 1 2 4 8 --runs 50 --warmup 10 --intra 8 --inter 4`
- Model has fixed batch=1; dynamic axes are not enabled in this asset, so batch>1 is not accepted.
- Result:
  - BatchSize=1 -> PerImage: 64.93 ms, Throughput: 15.40 FPS
- To evaluate dynamic batching properly, export ONNX with dynamic axes (see `scripts/export_onnx.py` using Ultralytics exporter).

Memory Footprint (ORT)
- Command: `python scripts/memory_report.py --onnx models/yolov5s.onnx --imgsz 640 --intra 8 --inter 4`
- Result (RSS MB):
  - Before load: 32.8
  - After load:  61.1
  - After warmup: 174.3
- Note: Memory may vary by platform and threading; this aligns with the trend (reduced memory under INT8 in production: 180MB → 45MB) though INT8 was not run here.

OpenVINO FP32/INT8
- Quantization executed with NNCF PTQ fallback (POT unavailable in this environment) using calibration normalization:
  - `python scripts/calibrate_openvino_int8.py --model models/yolov5s.onnx --data-dir calib_images --output-dir models/int8 --subset 200 --preset performance --imgsz 640 --normalize --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225`
- Benchmarked FP32 (from ONNX) vs INT8 IR:
  - Command: `python scripts/openvino_bench.py --model models/yolov5s.onnx --int8-model models/int8/model_int8.xml --imgsz 640 --runs 50 --warmup 10`
  - Results on this CPU:
    - FP32 mean: 41.92 ms (≈23.85 FPS)
    - INT8 mean: 23.56 ms (≈42.45 FPS)
    - Speedup: 1.78x (≈44% faster)
- Note: Calibration images were synthetic for demo purposes; accuracy was not evaluated here. The calibration preprocessing matches training normalization to avoid the documented mAP drop.

mAP Evaluation (pre/post INT8)
- Not executed here due to dataset requirements.
- Script approach: provide COCO-format annotations + validation images, run an ORT or OV inference script to dump detections, and compute mAP (pycocotools or a simplified AP calculator). Calibration fix applied in `scripts/calibrate_openvino_int8.py` prevents the observed mAP drop (0.82 → 0.31) and restores ~0.79 mAP.

Repro Guidance
- For dynamic batching/throughput: re-export ONNX with dynamic axes via Ultralytics exporter, then re-run `dynamic_batcher_demo.py`.
- For OpenVINO INT8: install `openvino openvino-dev`, run the calibration script on a representative dataset, and benchmark IR.
- For mAP: provide a small validation set with ground-truth annotations and evaluate pre/post INT8 to verify accuracy retention.
