#!/usr/bin/env python3
"""
OpenVINO POT INT8 calibration stub for YOLOv5 ONNX/IR models.

This script:
- Loads an ONNX or OpenVINO IR model.
- Builds a simple directory-based calibration DataLoader with letterbox + normalization.
- Runs DefaultQuantization (POT) targeting CPU.
- Saves INT8 IR to an output directory.

Example:
  python scripts/calibrate_openvino_int8.py \
    --model models/yolov5s.onnx --data-dir calib_images/ \
    --output-dir models/int8 --imgsz 640 --subset 300 --preset performance

Important:
- Ensure calibration preprocessing matches training-time normalization to preserve mAP:
  (img.astype(np.float32)/255.0 - 0.485) / 0.229 (per-channel mean/std is supported via args)
"""
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def lazy_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception as e:
        raise SystemExit("OpenCV (cv2) is required. pip install opencv-python") from e


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


def build_loader_files(data_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in data_dir.glob("**/*") if p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"No images found under {data_dir}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to ONNX or IR (.xml)")
    ap.add_argument("--data-dir", required=True, help="Directory with calibration images")
    ap.add_argument("--output-dir", required=True, help="Output directory for INT8 IR")
    ap.add_argument("--imgsz", type=int, default=640, help="Calibration image size")
    ap.add_argument("--subset", type=int, default=300, help="Number of images for statistics collection")
    ap.add_argument("--preset", choices=["performance", "mixed", "accuracy"], default="performance", help="POT preset")
    ap.add_argument("--device", default="CPU", help="Target device")
    ap.add_argument("--normalize", action="store_true", help="Apply mean/std normalization during calibration")
    ap.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406], help="Mean (RGB)")
    ap.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225], help="Std (RGB)")
    args = ap.parse_args()

    # Try POT first; if unavailable, fall back to NNCF PTQ
    pot_available = False
    try:
        from openvino.runtime import Core
        from openvino.tools.pot import DataLoader, IEEngine, load_model, save_model, create_pipeline, compress_model_weights  # type: ignore
        pot_available = True
    except Exception:
        from openvino.runtime import Core, serialize
        from nncf import Dataset
        from nncf.common.quantization.structs import QuantizationPreset
        from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core = Core()
    ov_model = core.read_model(model=str(model_path))
    if len(ov_model.inputs) != 1:
        raise SystemExit(f"Expected 1 input, found {len(ov_model.inputs)}")
    input_name = ov_model.inputs[0].get_any_name()

    # Build file list
    files = build_loader_files(data_dir)
    if args.subset > 0:
        files = files[: args.subset]

    if pot_available:
        # POT path
        pot_model = load_model(str(model_path))

        class DirLoader(DataLoader):
            def __init__(self, files, imgsz, input_name, normalize, mean, std):
                self.files = files
                self.imgsz = imgsz
                self.input_name = input_name
                self.normalize = normalize
                self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
                self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

            def __len__(self):
                return len(self.files)

            def __getitem__(self, index: int) -> Tuple[Any, Dict[str, np.ndarray]]:
                path = str(self.files[index])
                cv2 = lazy_import_cv2()
                im = cv2.imread(path, cv2.IMREAD_COLOR)
                if im is None:
                    raise RuntimeError(f"Failed to read {path}")
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = letterbox(im, new_shape=(self.imgsz, self.imgsz))
                img = im.astype(np.float32) / 255.0
                if self.normalize:
                    img = (img - self.mean) / self.std
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)  # 1x3xHxW
                return path, {self.input_name: img}

        loader = DirLoader(files, args.imgsz, input_name, args.normalize, args.mean, args.std)
        engine = IEEngine(
            config={"device": args.device, "stat_requests_number": 1, "eval_requests_number": 1},
            data_loader=loader,
            metric=None,
        )
        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": args.device,
                    "preset": args.preset,
                    "stat_subset_size": len(loader),
                },
            }
        ]
        pipeline = create_pipeline(algorithms, engine)
        compressed_model = pipeline.run(pot_model)
        compress_model_weights(compressed_model)
        save_model(compressed_model, str(out_dir), model_name="model_int8")
        print(f"Saved INT8 IR to: {out_dir}")
    else:
        # NNCF PTQ fallback: build dataset of input dicts
        def preprocess(path: str):
            cv2 = lazy_import_cv2()
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError(f"Failed to read {path}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = letterbox(im, new_shape=(args.imgsz, args.imgsz))
            img = im.astype(np.float32) / 255.0
            if args.normalize:
                mean = np.array(args.mean, dtype=np.float32).reshape(1, 1, 3)
                std = np.array(args.std, dtype=np.float32).reshape(1, 1, 3)
                img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            return {input_name: img}

        dataset = Dataset([str(p) for p in files], transform_func=preprocess)
        ap = AdvancedQuantizationParameters()
        from nncf import quantize
        q_model = quantize(
            ov_model,
            calibration_dataset=dataset,
            preset=QuantizationPreset.PERFORMANCE,
            advanced_parameters=ap,
            subset_size=len(files),
        )
        # Save INT8 IR
        out_xml = out_dir / "model_int8.xml"
        out_bin = out_dir / "model_int8.bin"
        serialize(q_model, str(out_xml), str(out_bin))
        print(f"Saved INT8 IR to: {out_xml}")


if __name__ == "__main__":
    main()
