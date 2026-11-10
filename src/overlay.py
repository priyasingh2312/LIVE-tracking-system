"""Shared on-picture label: a solid chip at the TOP-LEFT (just under the camera's own date/time stamp), used for the
live view and for the saved alert snapshots so both read the same."""
from __future__ import annotations

import cv2


def label_chip(frame, text: str, color) -> None:
    """Draw `text` (white, bold) on a solid chip of `color` (BGR) at the top-left, below the camera clock."""
    h, w = frame.shape[:2]
    sc = max(0.6, w / 1280 * 0.9)
    th = max(2, int(round(sc * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    while tw > w - 40 and sc > 0.4:                       # long text: shrink to fit the picture
        sc -= 0.05
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    x0, y0 = 8, int(0.065 * h)
    x1, y1 = x0 + tw + 28, y0 + tht + base + 18
    over = frame.copy()
    cv2.rectangle(over, (x0, y0), (x1, y1), tuple(int(c * 0.55) for c in color), -1)
    cv2.addWeighted(over, 0.93, frame, 0.07, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + 7, y1), color, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    org = (x0 + 18, y1 - 9 - base // 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th, cv2.LINE_AA)
