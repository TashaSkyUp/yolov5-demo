#!/usr/bin/env python3
"""
Generate a synthetic sample image for quick testing without external downloads.

Outputs: samples/sample.jpg (1280x720) with shapes and text.

Usage:
  python scripts/gen_sample_image.py
"""
from pathlib import Path


def main():
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as e:
        raise SystemExit("Requires numpy and opencv-python") from e

    w, h = 1280, 720
    img = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)

    # Road-like rectangle
    cv2.rectangle(img, (0, int(h * 0.65)), (w, h), (60, 60, 60), -1)
    # Lane lines
    for x in range(0, w, 80):
        cv2.rectangle(img, (x, int(h * 0.78)), (x + 40, int(h * 0.80)), (200, 200, 200), -1)

    # Colored boxes (simulated objects)
    cv2.rectangle(img, (200, 300), (330, 520), (0, 255, 255), 2)
    cv2.rectangle(img, (700, 260), (860, 520), (0, 140, 255), 3)
    cv2.circle(img, (1000, 360), 60, (0, 255, 0), 3)

    # Title text
    cv2.putText(img, "SAMPLE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (240, 240, 240), 3, cv2.LINE_AA)

    out_dir = Path("samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

