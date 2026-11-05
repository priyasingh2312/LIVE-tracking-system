"""Swappable video sources: webcam, RTSP/IP camera, or a video file.

The rest of the program only calls start() / read() / stop(), so moving from a
webcam to an IP camera is a config change, not a code change.
"""
from __future__ import annotations

import os
import platform
import threading
import time

import cv2


class FrameSource:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kind = cfg.get("type", "webcam").lower()
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._last_returned_id = -1
        self._running = False
        self._thread: threading.Thread | None = None
        self._eof = False

    # -- opening -----------------------------------------------------------
    def _open(self) -> bool:
        if self.kind == "webcam":
            idx = int(os.getenv("WEBCAM_INDEX") or self.cfg.get("webcam_index", 0))   # WEBCAM_INDEX: set by the dashboard
            # CAP_DSHOW opens faster / more reliably on Windows
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if self.cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg["width"]))
            if self.cfg.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg["height"]))
        elif self.kind == "rtsp":
            url = os.getenv("RTSP_URL") or self.cfg.get("rtsp_url", "")
            if not url:
                raise ValueError("RTSP source selected but no rtsp_url / RTSP_URL set")
            # TCP is more reliable than UDP over Wi-Fi / VPN
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.kind == "file":
            path = self.cfg.get("file_path", "")
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path!r}")
            cap = cv2.VideoCapture(path)
        else:
            raise ValueError(f"Unknown source type: {self.kind}")

        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "FrameSource":
        if not self._open():
            raise RuntimeError(f"Could not open video source ({self.kind}). Check the config.")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        """Reader thread: always keeps only the newest frame (no lag build-up).
        Files are read in order without dropping frames."""
        fps = (self._cap.get(cv2.CAP_PROP_FPS) or 25) if self.kind == "file" else 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self.kind == "file":
                    self._eof = True
                    return
                self._reconnect()
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            if self.kind == "file":
                # throttle so a file plays at roughly real-time speed
                time.sleep(1.0 / fps)

    def _reconnect(self):
        """Used for webcam/RTSP drops: retry with a short back-off."""
        print(f"[FrameSource] {self.kind} stream lost, reconnecting...")
        if self._cap is not None:
            self._cap.release()
        while self._running:
            time.sleep(2)
            try:
                if self._open():
                    print("[FrameSource] reconnected")
                    return
            except Exception as exc:  # noqa: BLE001
                print(f"[FrameSource] reconnect failed: {exc}")

    # -- API ---------------------------------------------------------------
    def read(self, timeout: float = 5.0):
        """Return (ok, frame) with the newest frame not returned before.
        ok=False means the file ended or the timeout expired."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_id != self._last_returned_id:
                    self._last_returned_id = self._frame_id
                    return True, self._frame.copy()
            if self._eof:
                return False, None
            time.sleep(0.005)
        return False, None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
