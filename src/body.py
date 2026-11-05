"""Body appearance embedding (person re-identification) with OpenCV's model zoo (YoutuReID, ~107 MB, 768 numbers).

Used ONLY to recognise a STRANGER again when their face is hidden. It never grants access: clothes change from day to
day and two people can dress alike, so authorisation always needs a face match.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.faces import ensure_model

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class BodyObs:
    quality: float                  # 0..1: size and whether the body is cut off by the frame edge
    feature: np.ndarray             # (768,) L2-normalised
    crop: np.ndarray                # BGR crop of the person (for saving)


class BodyEngine:
    def __init__(self, cfg: dict):
        mdir = cfg.get("models_dir", "models")
        self.min_h = int(cfg.get("min_body_h", 140))             # ignore people shorter than this many pixels
        self.net = cv2.dnn.readNetFromONNX(ensure_model("reid", mdir))

    def embed(self, frame, box) -> BodyObs | None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        h, w = y2 - y1, x2 - x1
        if h < self.min_h or w < 30 or h < 1.2 * w:                # too small, or not an upright person
            return None
        crop = frame[y1:y2, x1:x2]
        img = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
        x = ((img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        self.net.setInput(np.ascontiguousarray(x))
        f = self.net.forward().flatten().astype(np.float32)
        f /= (np.linalg.norm(f) + 1e-9)
        cut = (x1 <= 2) or (x2 >= W - 2) or (y2 >= H - 2) or (y1 <= 2)
        q = min(1.0, h / 300.0) * (0.6 if cut else 1.0)
        return BodyObs(float(q), f, crop.copy())
