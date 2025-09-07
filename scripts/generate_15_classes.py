#!/usr/bin/env python3
"""
Generate a random 15-class selection from labels/coco80.txt and write to labels/classes_15.txt.

Usage:
  python scripts/generate_15_classes.py
"""
import random
from pathlib import Path


def main():
    src = Path("labels/coco80.txt")
    dst = Path("labels/classes_15.txt")
    names = [l.strip() for l in src.read_text().splitlines() if l.strip()]
    sel = sorted(random.sample(names, 15))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(sel) + "\n")
    print("Wrote:", dst)
    print("Selected:", ", ".join(sel))


if __name__ == "__main__":
    main()

