"""Publishes the latest annotated frame and a small status file for the dashboard (dashboard/app.py).

Two files are written into data/live: frame.jpg (the picture shown in the window) and status.json. Each is written
to a temporary name and then renamed, so the dashboard never reads a half-written file. Nothing is sent over a
network; the dashboard reads the files from disk."""
from __future__ import annotations

import json
import os
import threading
import time

import cv2


class LiveFeed:
    def __init__(self, folder: str = "data/live", every_sec: float = 0.06, width: int = 960, quality: int = 75):
        self.folder = folder
        self.every = every_sec
        self.width = width
        self.quality = quality
        self._last = 0.0
        self._slot = None                                 # newest (frame, status): a worker thread encodes and writes it,
        self._cv = threading.Condition()                  # so the tracking loop never waits for JPEG encoding or the disk
        self._stop = False
        os.makedirs(folder, exist_ok=True)
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def publish(self, frame, status: dict, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < self.every:
            return
        self._last = now
        with self._cv:
            self._slot = (frame, dict(status, ts=now, running=True))      # an older, unwritten picture is simply replaced
            self._cv.notify()

    def _work(self) -> None:
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                frame, status = self._slot
                self._slot = None
            try:
                img = frame
                if self.width and frame.shape[1] > self.width:
                    f = self.width / frame.shape[1]
                    img = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                tmp = os.path.join(self.folder, "frame.tmp.jpg")
                cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                os.replace(tmp, os.path.join(self.folder, "frame.jpg"))
                self._write_status(status)
            except OSError:
                pass                                      # the dashboard is optional: never stop tracking for it

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._slot = None
            self._cv.notify()
        self._worker.join(timeout=2)
        try:
            os.remove(os.path.join(self.folder, "frame.jpg"))      # never leave a stale annotated picture behind
        except OSError:
            pass
        try:
            self._write_status({"ts": time.time(), "running": False, "objects": []})
        except OSError:
            pass

    def _write_status(self, status: dict) -> None:
        tmp = os.path.join(self.folder, "status.tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f)
        os.replace(tmp, os.path.join(self.folder, "status.json"))
