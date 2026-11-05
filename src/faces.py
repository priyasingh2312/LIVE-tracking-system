"""Face detection + embedding using OpenCV's built-in models (no extra pip packages):
  * YuNet  - face detector (+5 landmarks)
  * SFace  - face recogniser (128-d embedding, cosine similarity; ~0.36 is the usual same-person cut-off)
The two ONNX files are downloaded once into ./models on first use.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MODELS = {
    "vittrack": ("object_tracking_vittrack_2023sep.onnx", [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
    ]),
    "reid": ("person_reid_youtu_2021nov.onnx", [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx",
    ]),
    "yunet": ("face_detection_yunet_2023mar.onnx", [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ]),
    "sface": ("face_recognition_sface_2021dec.onnx", [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ]),
}


def ensure_model(key: str, models_dir: str) -> str:
    fname, urls = MODELS[key]
    path = os.path.join(models_dir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    os.makedirs(models_dir, exist_ok=True)
    last = None
    for url in urls:
        try:
            print(f"Downloading {fname} ... (one time)")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < 100_000:   # git-LFS pointer instead of the real file
                raise RuntimeError("downloaded file is too small")
            os.replace(tmp, path)
            return path
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(
        f"Could not download {fname} ({last}).\n"
        f"Download it manually from {urls[-1]} and put it in the '{models_dir}' folder.")


@dataclass
class FaceObs:
    box: tuple[int, int, int, int]      # x1, y1, x2, y2 in full-frame pixels
    det_score: float
    quality: float                      # 0..1: size, sharpness, frontal-ness, detector score
    feature: np.ndarray                 # (128,) L2-normalised
    crop: np.ndarray                    # BGR face crop with margin (for saving)


class FaceEngine:
    def __init__(self, cfg: dict):
        mdir = cfg.get("models_dir", "models")
        self.det_thresh = float(cfg.get("det_thresh", 0.7))
        self.min_face_px = int(cfg.get("min_face_px", 60))
        self.det = cv2.FaceDetectorYN.create(ensure_model("yunet", mdir), "", (320, 320),
                                             self.det_thresh, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(ensure_model("sface", mdir), "")

    def detect_embed(self, frame, region=None, min_face_px: int | None = None) -> list[FaceObs]:
        """Find faces inside `region` (x1,y1,x2,y2) of the frame (default: whole frame)."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = region if region else (0, 0, W, H)
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return []
        crop = frame[y1:y2, x1:x2]
        scale = 1.0
        if crop.shape[1] < 320:                       # small crop: upscale so YuNet can see small faces
            scale = min(2.0, 320.0 / crop.shape[1])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = crop.shape[:2]
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(crop)
        out: list[FaceObs] = []
        if rows is None:
            return out
        min_px = self.min_face_px if min_face_px is None else min_face_px
        for row in rows:
            fx, fy, fw, fh = row[:4]
            face_px = fw / scale
            if face_px < min_px:
                continue
            feat = self.rec.feature(self.rec.alignCrop(crop, row)).flatten().astype(np.float32)
            feat /= (np.linalg.norm(feat) + 1e-9)
            # crop with margin for saving
            m = 0.3
            cx1, cy1 = max(0, int(fx - m * fw)), max(0, int(fy - m * fh))
            cx2, cy2 = min(w, int(fx + fw * (1 + m))), min(h, int(fy + fh * (1 + m)))
            face_img = crop[cy1:cy2, cx1:cx2].copy()
            q = self._quality(row, face_img, face_px)
            box = (int(x1 + fx / scale), int(y1 + fy / scale),
                   int(x1 + (fx + fw) / scale), int(y1 + (fy + fh) / scale))
            out.append(FaceObs(box, float(row[14]), q, feat, face_img))
        return out

    @staticmethod
    def _quality(row, face_img, face_px) -> float:
        q_size = min(1.0, face_px / 120.0)
        gray = cv2.cvtColor(cv2.resize(face_img, (96, 96)), cv2.COLOR_BGR2GRAY)
        q_sharp = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0)
        # landmarks: right eye, left eye, nose, mouth right, mouth left (image coordinates)
        re, le, nose = row[4:6], row[6:8], row[8:10]
        eye_d = max(1.0, abs(le[0] - re[0]))
        off = abs(nose[0] - (re[0] + le[0]) / 2) / eye_d
        q_front = max(0.0, 1.0 - off / 0.5)
        return float(0.35 * q_size + 0.25 * q_sharp + 0.25 * q_front + 0.15 * min(1.0, row[14]))
