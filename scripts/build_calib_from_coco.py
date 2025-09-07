#!/usr/bin/env python3
"""
Build a calibration image subset containing only the selected classes (e.g., the 15 picked from COCO).

Inputs (COCO format):
- --images-dir: Directory containing COCO images (e.g., val2017/)
- --ann: COCO annotations JSON (e.g., instances_val2017.json)
- --classes-file: File with allowed class names (default: labels/classes_15.txt)
- --out: Output directory to copy images into (default: calib_images_15/)
- --per-class: Target images per class (default: 20)

Strategy:
- Map names -> category_ids from the COCO JSON
- Collect image_ids per selected category
- Balance-select images to cover each class up to --per-class (without duplication)
- Copy images into --out

Usage example:
  python scripts/build_calib_from_coco.py \
    --images-dir /data/coco/val2017 \
    --ann /data/coco/annotations/instances_val2017.json \
    --classes-file labels/classes_15.txt \
    --out calib_images_15 --per-class 20
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="COCO images directory (e.g., val2017)")
    ap.add_argument("--ann", required=True, help="COCO instances_*.json path")
    ap.add_argument("--classes-file", default="labels/classes_15.txt", help="Allowed class names file")
    ap.add_argument("--out", default="calib_images_15", help="Output directory for calibration images")
    ap.add_argument("--per-class", type=int, default=20, help="Target images per class")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ann, "r") as f:
        coco = json.load(f)

    allowed_names = [l.strip() for l in Path(args.classes_file).read_text().splitlines() if l.strip()]

    # Map category id -> name and name -> id
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

    missing = [n for n in allowed_names if n not in name_to_cat_id]
    if missing:
        raise SystemExit(f"Classes not found in annotations: {missing}")

    allowed_cat_ids = [name_to_cat_id[n] for n in allowed_names]

    # Build image_id -> file_name
    image_id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}

    # Collect image ids per category
    imgs_by_cat = defaultdict(set)
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid in allowed_cat_ids:
            imgs_by_cat[cid].add(ann["image_id"])

    # Balanced selection: ensure up to per-class images per category (no duplicates)
    selected = set()
    counts = defaultdict(int)
    # Pool: union of all image ids that contain at least one allowed category
    pool = set().union(*imgs_by_cat.values()) if imgs_by_cat else set()
    pool = list(pool)
    random.shuffle(pool)

    # Iterate pool, add images that help fill per-class quota
    for img_id in pool:
        # categories present for this image from allowed set
        present = [cid for cid in allowed_cat_ids if img_id in imgs_by_cat.get(cid, set())]
        if not present:
            continue
        # Check if this image helps any class still under quota
        if any(counts[cid] < args.per_class for cid in present):
            selected.add(img_id)
            for cid in present:
                if counts[cid] < args.per_class:
                    counts[cid] += 1
        # Stop when all classes meet quota
        if all(counts[cid] >= args.per_class for cid in allowed_cat_ids):
            break

    print("Selected images:", len(selected))
    print("Per-class coverage:")
    for cid in allowed_cat_ids:
        print(f"- {cat_id_to_name[cid]}: {counts[cid]}")

    # Copy files
    copied = 0
    for img_id in selected:
        fname = image_id_to_fname[img_id]
        src = images_dir / fname
        dst = out_dir / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"WARNING: missing image: {src}")
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} images to {out_dir}")


if __name__ == "__main__":
    main()

