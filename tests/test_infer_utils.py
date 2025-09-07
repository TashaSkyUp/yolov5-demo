#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module(module_path):
    spec = importlib.util.spec_from_file_location("infer_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


INFER_PATH = str(Path(__file__).resolve().parents[1] / "scripts" / "infer_onnx.py")
infer_mod = load_module(INFER_PATH)


class TestPostprocessAndNMS(unittest.TestCase):
    def test_box_iou(self):
        a = np.array([[0, 0, 2, 2], [0, 0, 1, 1]], dtype=np.float32)
        b = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
        iou = infer_mod.box_iou(a, b)
        # Expected IoU matrix:
        # a0 with b0: 0.25, a0 with b1: 0.25
        # a1 with b0: 1.0,  a1 with b1: 0.0
        self.assertAlmostEqual(iou[0, 0], 0.25, places=5)
        self.assertAlmostEqual(iou[0, 1], 0.25, places=5)
        self.assertAlmostEqual(iou[1, 0], 1.0, places=5)
        self.assertAlmostEqual(iou[1, 1], 0.0, places=5)

    def test_nms(self):
        boxes = np.array([
            [0, 0, 2, 2],  # high score
            [0.5, 0.5, 2.5, 2.5],  # overlaps with first
            [3, 3, 4, 4],  # separate
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        # With a lower IoU threshold, overlapping box should be suppressed
        keep = infer_mod.nms(boxes, scores, iou_thres=0.3)
        self.assertEqual(set(keep.tolist()), {0, 2})

    def test_postprocess(self):
        # Create two predictions for a single image: one strong, one weak overlapping
        # pred format: [cx, cy, w, h, obj, cls0, cls1]
        pred = np.array([
            [50, 50, 40, 40, 0.9, 0.1, 0.8],  # strong class1
            [52, 52, 42, 42, 0.8, 0.2, 0.6],  # overlapping, lower score
            [150, 150, 20, 20, 0.7, 0.7, 0.2],  # separate class0
        ], dtype=np.float32)
        h0, w0 = 200, 200
        pad = (0.0, 0.0)
        scale = 1.0
        dets = infer_mod.postprocess(pred, (h0, w0), pad, scale, conf_thres=0.25, iou_thres=0.5, max_det=300)
        # Expect 2 final detections after NMS
        self.assertEqual(len(dets), 2)
        # Highest score should be first box (class 1)
        boxes, scores, classes = zip(*dets)
        self.assertIn(int(classes[0]), (0, 1))
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
