#!/usr/bin/env python3
"""
ONNX Runtime CPU inference for YOLOv5 ONNX models with threading options and NMS.

Example:
  python scripts/infer_onnx.py --onnx models/yolov5s.onnx --image sample.jpg \
    --conf 0.25 --iou 0.45 --intra 8 --inter 4 --save-vis outputs

Notes:
- Assumes ONNX export returns (N, 25200, 5+num_classes) per batch item (typical YOLOv5 export).
- Preprocessing: letterbox to imgsz, BGR->RGB, [0,1] scaling; optional mean/std normalization.
"""
import argparse
import time
from pathlib import Path

import numpy as np


def lazy_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception as e:
        raise SystemExit("OpenCV (cv2) is required. pip install opencv-python") from e


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if scaleFill:  # stretch
        new_unpad = (new_shape[1], new_shape[0])
        dw, dh = 0.0, 0.0
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    cv2 = lazy_import_cv2()
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def box_iou(box1, box2):
    # box1: (N,4) xyxy, box2: (M,4)
    # Promote to float32 to avoid fp16 overflows and NaNs
    box1 = box1.astype(np.float32, copy=False)
    box2 = box2.astype(np.float32, copy=False)

    def area(b):
        w = (b[:, 2] - b[:, 0]).clip(0)
        h = (b[:, 3] - b[:, 1]).clip(0)
        return w * h

    iw = np.maximum(0.0, np.minimum(box1[:, None, 2], box2[None, :, 2]) - np.maximum(box1[:, None, 0], box2[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(box1[:, None, 3], box2[None, :, 3]) - np.maximum(box1[:, None, 1], box2[None, :, 1]))
    inter = iw * ih
    denom = area(box1)[:, None] + area(box2)[None, :] - inter + 1e-6
    iou = inter / denom
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def nms(boxes, scores, iou_thres=0.45, max_det=300):
    # boxes: (N,4) xyxy, scores: (N,)
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < max_det:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int64)


def postprocess(pred, img_shape, pad, scale, conf_thres=0.25, iou_thres=0.45, max_det=300):
    # pred: (N, 85) -> xywh + obj + cls_scores
    # Convert to xyxy on original image space and apply NMS
    pred = pred.astype(np.float32, copy=False)
    xywh = pred[:, :4]
    obj = pred[:, 4:5]
    cls = pred[:, 5:]
    if cls.size == 0:
        return []
    cls_idx = cls.argmax(1)
    cls_score = cls[np.arange(cls.shape[0]), cls_idx]
    scores = (obj[:, 0] * cls_score)
    mask = scores > conf_thres
    if not mask.any():
        return []
    xywh = xywh[mask]
    scores = scores[mask]
    cls_idx = cls_idx[mask]

    # xywh -> xyxy on letterboxed image
    x, y, w, h = xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32, copy=False)

    # Undo letterbox scaling
    (h0, w0) = img_shape
    dw, dh = pad
    gain = scale
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, [0, 2]] /= gain
    boxes[:, [1, 3]] /= gain

    # Clip and ensure finite
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    boxes = np.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)

    # NMS
    keep = nms(boxes, scores, iou_thres=iou_thres, max_det=max_det)
    boxes = boxes[keep]
    scores = scores[keep]
    cls_idx = cls_idx[keep]
    return [(boxes[i], float(scores[i]), int(cls_idx[i])) for i in range(len(keep))]


def draw_detections(img, dets, class_names=None, color=(0, 255, 0)):
    cv2 = lazy_import_cv2()
    img = img.copy()
    for (x1, y1, x2, y2), score, cls in dets:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{cls}:{score:.2f}"
        if class_names and 0 <= cls < len(class_names):
            label = f"{class_names[cls]}:{score:.2f}"
        cv2.putText(img, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def load_images(args):
    cv2 = lazy_import_cv2()
    paths = []
    if args.image:
        paths = [args.image]
    elif args.dir:
        img_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [str(p) for p in Path(args.dir).glob("**/*") if p.suffix.lower() in img_ext]
    if not paths:
        raise SystemExit("No images found. Use --image or --dir.")
    images = []
    metas = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h0, w0 = im.shape[:2]
        lb, r, (dw, dh) = letterbox(im, new_shape=(args.imgsz, args.imgsz))
        lb_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        img = lb_rgb.astype(np.float32) / 255.0
        if args.normalize:
            mean = np.array(args.mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(args.std, dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC->CHW
        images.append(img)
        metas.append(((h0, w0), r, (dw, dh), p))
    return np.stack(images, axis=0), metas


def load_allowed_classes(path):
    if not path:
        return None, None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Allowed classes file not found: {p}")
    allowed = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    coco_path = Path("labels/coco80.txt")
    if coco_path.exists():
        all_names = [l.strip() for l in coco_path.read_text().splitlines() if l.strip()]
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        idxs = [name_to_idx[n] for n in allowed if n in name_to_idx]
    else:
        idxs = [int(x) for x in allowed]
        all_names = None
    return idxs, all_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Single image path")
    g.add_argument("--dir", help="Directory of images")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for inference")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing")
    ap.add_argument("--intra", type=int, default=8, help="ONNX Runtime intra-op threads")
    ap.add_argument("--inter", type=int, default=4, help="ONNX Runtime inter-op threads")
    ap.add_argument("--normalize", action="store_true", help="Apply mean/std normalization")
    ap.add_argument("--mean", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Mean for normalization (RGB)")
    ap.add_argument("--std", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Std for normalization (RGB)")
    ap.add_argument("--save-vis", default=None, help="Output directory to save visualizations")
    ap.add_argument("--allowed-classes-file", default=None, help="Path to allowed class names (one per line), e.g., labels/classes_15.txt")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise SystemExit("onnxruntime is required. pip install onnxruntime") from e

    images, metas = load_images(args)
    n = images.shape[0]

    # Build session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra
    sess_options.inter_op_num_threads = args.inter
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_type = sess.get_inputs()[0].type or "tensor(float)"
    expected_dtype = np.float16 if "float16" in input_type else np.float32
    allowed_idxs, all_names = load_allowed_classes(args.allowed_classes_file)

    # Warmup
    dummy = np.zeros((args.batch, 3, args.imgsz, args.imgsz), dtype=expected_dtype)
    for _ in range(args.warmup):
        _ = sess.run([output_name], {input_name: dummy})

    # Inference loop
    t0 = time.time()
    all_dets = []
    for i in range(0, n, args.batch):
        batch = images[i : i + args.batch].astype(expected_dtype)
        start = time.time()
        out = sess.run([output_name], {input_name: batch})[0]
        end = time.time()
        # out: (B, N, 5+nc)
        for b in range(out.shape[0]):
            pred = out[b]
            (h0, w0), r, (dw, dh), path = metas[i + b]
            dets = postprocess(pred, (h0, w0), (dw, dh), r, conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det)
            if allowed_idxs is not None:
                dets = [d for d in dets if d[2] in allowed_idxs]
            all_dets.append((path, dets, end - start))

    t1 = time.time()
    total_time = t1 - t0
    num_images = len(all_dets)
    print(f"Processed {num_images} images in {total_time:.3f}s -> {1000*total_time/num_images:.1f} ms/img")

    # Save visualizations
    if args.save_vis:
        out_dir = Path(args.save_vis)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2 = lazy_import_cv2()
        for (path, dets, _lat) in all_dets:
            im0 = cv2.imread(path, cv2.IMREAD_COLOR)
            vis = draw_detections(im0, dets, class_names=all_names)
            out_path = out_dir / (Path(path).stem + "_pred.jpg")
            cv2.imwrite(str(out_path), vis)
        print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
