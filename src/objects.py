"""Remembers the objects you outline, so the dashboard's Objects page can show them.

Every object you outline at Start gets an id and a folder data/objects/<id>/ holding
  crop.jpg   the cut-out of the object (the tile picture)
  scene.jpg  the whole frame it was outlined on, with the outline drawn
  run_*.jpg  one more cut-out each time a saved object is reused (newest 8 are kept)
and a row in the 'objects' table of the faces database (camera, dates, how often it was used).

The tracker keeps using a fixed label ("object 5") in the pickup log; the dashboard shows the name you give it.
Nothing here is used for tracking decisions, and a failure here never stops the tracker."""
from __future__ import annotations

import os
import sqlite3
import time

import cv2
import numpy as np

MAX_RUNS = 8


def _crop(frame, box, pad=0.12):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    c = frame[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
    return c if c.size else None


def _scene(frame, items, mine, max_w=960):
    img = frame.copy()
    for it in items:
        col = (0, 200, 0) if it is mine else (200, 200, 0)
        pts = it.get("poly")
        if pts and len(pts) > 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, col, 3 if it is mine else 2)
        else:
            x1, y1, x2, y2 = it["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if it is mine else 2)
    if img.shape[1] > max_w:
        f = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def register_objects(frame, items, db_path: str, data_dir: str) -> list:
    """Save pictures + a database row for each item. Items that carry an 'oid' of a saved object are reused (their
    use count goes up and a new cut-out is added); all others become new objects. Sets item['oid'] and item['name']
    (the fixed label) and returns the items."""
    camera = (os.environ.get("CAMERA_ID", "") + " " + os.environ.get("CAMERA_NAME", "")).strip() or "terminal"
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS objects(id INTEGER PRIMARY KEY, label TEXT, name TEXT, camera TEXT, "
                    "created REAL, last_used REAL, times_used INTEGER)")
        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        for it in items:
            row = con.execute("SELECT id, label FROM objects WHERE id=?", (it.get("oid"),)).fetchone() if it.get("oid") else None
            crop = _crop(frame, it["box"])
            if row:
                oid, label = row
                con.execute("UPDATE objects SET last_used=?, times_used=COALESCE(times_used,0)+1, camera=? WHERE id=?",
                            (now, camera, oid))
                folder = os.path.join(data_dir, "objects", str(oid))
                os.makedirs(folder, exist_ok=True)
                if crop is not None:
                    cv2.imwrite(os.path.join(folder, f"run_{stamp}.jpg"), crop)
                    runs = sorted(f for f in os.listdir(folder) if f.startswith("run_"))
                    for old in runs[:-MAX_RUNS]:
                        os.remove(os.path.join(folder, old))
            else:
                cur = con.execute("INSERT INTO objects(label, name, camera, created, last_used, times_used) VALUES('', '', ?,?,?,1)",
                                  (camera, now, now))
                oid = cur.lastrowid
                label = f"object {oid}"
                con.execute("UPDATE objects SET label=?, name=? WHERE id=?", (label, label, oid))
                folder = os.path.join(data_dir, "objects", str(oid))
                os.makedirs(folder, exist_ok=True)
                if crop is not None:
                    cv2.imwrite(os.path.join(folder, "crop.jpg"), crop)
                cv2.imwrite(os.path.join(folder, "scene.jpg"), _scene(frame, items, it), [cv2.IMWRITE_JPEG_QUALITY, 80])
            it["oid"], it["name"] = oid, label
        con.commit()
    finally:
        con.close()
    return items
