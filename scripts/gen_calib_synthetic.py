#!/usr/bin/env python3
"""
Generate a synthetic calibration image set (placeholder) under a target directory.

This is a stopgap to keep a calibration folder tracked in git. Replace with a
real, representative set (e.g., via build_calib_from_coco.py) before accuracy eval.

Usage:
  python scripts/gen_calib_synthetic.py --out calib_images_15 --count 60 --imgsz 640
"""
import argparse
from pathlib import Path
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib_images_15")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as e:
        raise SystemExit("Requires numpy and opencv-python") from e

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        # gradient background
        for y in range(args.imgsz):
            c = int(255 * y / max(1, args.imgsz - 1))
            img[y, :, :] = (c // 2, c, (255 - c))
        # random rectangles/circles/lines
        for k in range(10):
            color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
            if k % 3 == 0:
                x1, y1 = np.random.randint(0, args.imgsz - 50, size=2)
                x2, y2 = x1 + np.random.randint(20, 200), y1 + np.random.randint(20, 200)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            elif k % 3 == 1:
                x, y = np.random.randint(50, args.imgsz - 50, size=2)
                r = np.random.randint(10, 80)
                cv2.circle(img, (int(x), int(y)), int(r), color, 2)
            else:
                x1, y1 = np.random.randint(0, args.imgsz, size=2)
                x2, y2 = np.random.randint(0, args.imgsz, size=2)
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(img, f"calib {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"img_{i:04d}.jpg"), img)

    print(f"Wrote {args.count} synthetic calibration images to {out}")


if __name__ == "__main__":
    main()

