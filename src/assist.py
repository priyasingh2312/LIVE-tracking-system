"""Detector assist: a standard YOLO object detector that helps the generic tracker when the selected object is a
common kind of thing (bottle, phone, cup, bag, laptop ...).

The generic tracker follows ANY object but is weak on small, dark or thin things. If the object the client selected
turns out to be one of the 80 standard classes, this module finds the class once (from the frozen start frame) and
afterwards offers candidate boxes of that class, so the follower can re-find the object quickly (in a hand, or put
down somewhere new) or snap back after a drift. Objects that are not a standard class are simply not assisted."""
from __future__ import annotations

import numpy as np

try:
    from ultralytics import YOLO
except Exception:                                   # ultralytics missing: assist is simply off
    YOLO = None


def _overlap(a, b):
    """Share of box `a` that lies inside box `b`, and IoU."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa, bb = max(1, (a[2] - a[0]) * (a[3] - a[1])), max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / aa, inter / (aa + bb - inter)


class ObjectAssist:
    def __init__(self, cfg: dict, device: str, ref_frame, items: list[dict]):
        self.model = YOLO(cfg.get("model", "yolo11n.pt"))
        self.device = device
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.15))
        self.names = self.model.names
        self.cls: dict[int, int | None] = {}          # object number -> COCO class id (or None)
        res = self.model.predict(ref_frame, conf=0.08, imgsz=self.imgsz, device=device, verbose=False)[0]
        boxes = [] if res.boxes is None else [(tuple(int(v) for v in b.xyxy[0].tolist()), int(b.cls[0]), float(b.conf[0]))
                                              for b in res.boxes]
        for i, it in enumerate(items):
            best, cid = 0.0, None
            for bx, c, cf in boxes:
                inside, iou = _overlap(tuple(it["box"]), bx)
                score = max(iou, 0.7 * inside) * (0.5 + cf)
                if max(iou, inside) >= 0.3 and score > best:
                    best, cid = score, c
            self.cls[i + 1] = cid
            if cid is None:
                print(f"Assist: '{it['name']}' is not a standard object type; generic tracking only.")
            else:
                print(f"Assist: '{it['name']}' looks like a '{self.names[cid]}'; the detector will help the tracker.")
        self._cache_idx, self._cache = -1, {}

    def wants(self, idx: int) -> bool:
        return self.cls.get(idx) is not None

    def boxes(self, frame, frame_idx: int, idx: int) -> list[tuple[int, int, int, int]]:
        """Boxes (x1,y1,x2,y2) of the selected object's class in this frame (one detector run per frame)."""
        cid = self.cls.get(idx)
        if cid is None:
            return []
        if self._cache_idx != frame_idx:
            ids = sorted({c for c in self.cls.values() if c is not None})
            res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                     classes=ids, verbose=False)[0]
            self._cache = {}
            if res.boxes is not None:
                for b in res.boxes:
                    self._cache.setdefault(int(b.cls[0]), []).append(tuple(int(v) for v in b.xyxy[0].tolist()))
            self._cache_idx = frame_idx
        return self._cache.get(cid, [])
