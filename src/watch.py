"""Zone watcher: detects that a user-selected object has been removed from its spot,
and which tracked person's hand was at the spot just before.

Works for ANY object (no training). Assumes a fixed camera and an object that sits still.
How it decides an object is gone:
  * appearance: the patch inside the selected box is compared with a reference patch taken at
    selection time (normalised cross-correlation + mean pixel difference, small shift allowed);
  * hand: a person's wrist inside the (expanded) zone marks the zone as "occupied" and records
    that person as the latest toucher; removal is never decided while a hand is in the zone;
  * once the hand is gone and the patch still does not match for `absent_frames` processed
    frames, the object counts as removed and the latest toucher is blamed.
If nobody's hand was seen, the alert is slower and is reported as "missing" (no known person).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.detector import Detection


@dataclass
class WatchEvent:
    ts: float
    kind: str                       # "pickup" | "missing" | "returned"
    item: str
    item_box: tuple[int, int, int, int]
    person_id: int | None = None
    person_box: tuple[int, int, int, int] | None = None


@dataclass
class _Item:
    name: str
    box: tuple[int, int, int, int]
    ref: np.ndarray
    scale: float
    state: str = "present"          # present | removed
    occupied: bool = False
    score: float = 1.0
    gone: int = 0
    back: int = 0
    wrist_contact: dict = field(default_factory=dict)   # person_id -> (ts, person_box)
    box_contact: dict = field(default_factory=dict)


class ZoneWatcher:
    def __init__(self, cfg: dict, items: list[dict], ref_frame):
        self.ncc_thresh = float(cfg.get("ncc_thresh", 0.5))
        self.mad_thresh = float(cfg.get("mad_thresh", 12))
        self.absent_frames = int(cfg.get("absent_frames", 6))
        self.return_frames = int(cfg.get("return_frames", 10))
        self.touch_margin = float(cfg.get("touch_margin", 0.5))
        self.box_overlap = float(cfg.get("box_overlap", 0.5))
        self.contact_window = float(cfg.get("contact_window_sec", 6.0))
        gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        self.items: list[_Item] = []
        for d in items:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            crop = gray[y1:y2, x1:x2]
            scale = min(1.0, 96.0 / max(x2 - x1, y2 - y1))
            self.items.append(_Item(d["name"], (x1, y1, x2, y2), self._prep(crop, scale), scale))

    @staticmethod
    def _prep(img, scale):
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _match(self, gray, it: _Item):
        x1, y1, x2, y2 = it.box
        h, w = gray.shape
        pad = int(0.12 * max(x2 - x1, y2 - y1))
        search = gray[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        search = self._prep(search, it.scale)
        th, tw = it.ref.shape
        dh, dw = max(0, th - search.shape[0]), max(0, tw - search.shape[1])
        if dh or dw:
            search = cv2.copyMakeBorder(search, 0, dh, 0, dw, cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(search, it.ref, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        _, best, _, loc = cv2.minMaxLoc(res)
        patch = search[loc[1]:loc[1] + th, loc[0]:loc[0] + tw]
        mad = float(np.mean(np.abs(patch.astype(np.int16) - it.ref.astype(np.int16))))
        return float(best), mad, patch

    def _latest(self, contacts: dict, now: float):
        best = None
        for pid, (ts, box) in contacts.items():
            if now - ts <= self.contact_window and (best is None or ts > best[1]):
                best = (pid, ts, box)
        return (best[0], best[2]) if best else None

    def update(self, frame, persons: list[Detection], now: float | None = None) -> list[WatchEvent]:
        now = time.time() if now is None else now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        events: list[WatchEvent] = []
        for it in self.items:
            ncc, mad, patch = self._match(gray, it)
            it.score = ncc
            present = ncc >= self.ncc_thresh or mad <= self.mad_thresh

            x1, y1, x2, y2 = it.box
            m = self.touch_margin * max(x2 - x1, y2 - y1)
            ex1, ey1, ex2, ey2 = x1 - m, y1 - m, x2 + m, y2 + m
            area = max(1, (x2 - x1) * (y2 - y1))
            occupied = False
            for p in persons:
                if p.track_id is None:
                    continue
                for wx, wy in p.wrists:
                    if ex1 <= wx <= ex2 and ey1 <= wy <= ey2:
                        occupied = True
                        it.wrist_contact[p.track_id] = (now, p.box)
                bx1, by1, bx2, by2 = p.box
                iw = max(0, min(x2, bx2) - max(x1, bx1))
                ih = max(0, min(y2, by2) - max(y1, by1))
                if iw * ih / area >= self.box_overlap:
                    it.box_contact[p.track_id] = (now, p.box)
            it.occupied = occupied

            if it.state == "present":
                if present:
                    it.gone = 0
                    if not occupied and ncc >= self.ncc_thresh:   # follow slow lighting drift
                        it.ref = cv2.addWeighted(it.ref, 0.98, patch, 0.02, 0)
                elif not occupied:
                    it.gone += 1
                    by_hand = self._latest(it.wrist_contact, now)
                    by_body = None if by_hand else self._latest(it.box_contact, now)
                    need = self.absent_frames if by_hand else self.absent_frames * 3
                    if it.gone >= need:
                        it.state, it.back = "removed", 0
                        who = by_hand or by_body
                        events.append(WatchEvent(now, "pickup" if by_hand else "missing", it.name, it.box,
                                                 who[0] if who else None, who[1] if who else None))
            else:  # removed
                if present and not occupied:
                    it.back += 1
                    if it.back >= self.return_frames:
                        it.state, it.gone = "present", 0
                        it.ref = patch.copy()
                        events.append(WatchEvent(now, "returned", it.name, it.box))
                else:
                    it.back = 0
        return events
