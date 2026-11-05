"""YOLO detection + tracking wrapper.

Used twice: once with a pose model for people (gives wrist keypoints) and once
with a normal detection model for objects such as cell phones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ultralytics import YOLO

LEFT_WRIST, RIGHT_WRIST = 9, 10  # COCO keypoint indices


@dataclass
class Detection:
    track_id: int | None
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    wrists: list[tuple[float, float]] = field(default_factory=list)  # pose models only

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


class Detector:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.model = YOLO(cfg["model"])
        self.device = device
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.35))
        self.tracker = cfg.get("tracker", "bytetrack.yaml")
        self.kp_conf = float(cfg.get("kp_conf", 0.4))

        names = self.model.names  # {id: name}
        wanted = set(cfg.get("classes", ["person"]))
        self.class_ids = [i for i, n in names.items() if n in wanted]
        unknown = wanted - {names[i] for i in self.class_ids}
        if unknown:
            raise ValueError(f"Classes not in model {cfg['model']}: {sorted(unknown)}")

    def track(self, frame) -> list[Detection]:
        res = self.model.track(
            frame,
            persist=True,              # keep track IDs between calls
            tracker=self.tracker,
            classes=self.class_ids,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        n = len(res.boxes)
        ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * n
        kp_xy = kp_cf = None
        if res.keypoints is not None and res.keypoints.xy is not None:
            kp_xy = res.keypoints.xy.tolist()
            kp_cf = res.keypoints.conf.tolist() if res.keypoints.conf is not None else None

        for i in range(n):
            x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i].tolist())
            wrists: list[tuple[float, float]] = []
            if kp_xy is not None:
                for k in (LEFT_WRIST, RIGHT_WRIST):
                    px, py = kp_xy[i][k]
                    c = kp_cf[i][k] if kp_cf is not None else 1.0
                    if c >= self.kp_conf and (px > 0 or py > 0):
                        wrists.append((px, py))
            out.append(Detection(ids[i], self.model.names[int(res.boxes.cls[i])],
                                 float(res.boxes.conf[i]), (x1, y1, x2, y2), wrists))
        return out
