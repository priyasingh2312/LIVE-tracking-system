"""Runtime object selection: freeze a camera frame, outline each object with a polygon to protect/follow.
Large IP-camera frames are shown scaled down (max_width) and the boxes are converted back to full size."""
from __future__ import annotations

import json
import os

import cv2

WINDOW = "select objects"


def _fit(img, max_w):
    s = min(1.0, max_w / img.shape[1])
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else img), s


def grab_frame(source, max_w=1280):
    """Show the live feed; press SPACE to freeze the current frame."""
    cv2.namedWindow(WINDOW)
    print("Live preview: put the objects in place, then press SPACE to freeze a frame (q = quit).")
    while True:
        ok, frame = source.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the video source.")
        disp, _ = _fit(frame, max_w)
        disp = disp.copy()
        cv2.putText(disp, "Press SPACE to freeze this frame  (q = quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return frame
        if key in (27, ord("q")):
            raise SystemExit("Cancelled.")


def poly_item(points, name, shape):
    """Build an item from polygon points (full-frame pixels): the polygon + its bounding rectangle.
    The tracker needs a rectangle, so 'box' is the polygon's bounding box; 'poly' keeps the exact outline."""
    H, W = shape[:2]
    pts = [[min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1)] for x, y in points]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return {"name": name, "box": [x1, y1, x2, y2], "poly": pts}


def _draw_poly(img, pts, color, closed, s=1.0):
    if not pts:
        return
    arr = [(int(x * s), int(y * s)) for x, y in pts]
    for a, b in zip(arr, arr[1:]):
        cv2.line(img, a, b, color, 2)
    if closed and len(arr) > 2:
        cv2.line(img, arr[-1], arr[0], color, 2)
    for a in arr:
        cv2.circle(img, a, 4, color, -1)


def select_items(frame, max_w=1280) -> list[dict]:
    """Outline each object with a polygon: LEFT-CLICK to place corners, then ENTER (or RIGHT-CLICK) to close it.
    BACKSPACE/U = undo last corner, ESC = clear the current outline, C = finish (ESC with nothing drawn also finishes)."""
    small, s = _fit(frame, max_w)
    items: list[dict] = []
    pts: list[tuple[int, int]] = []          # current polygon, display coordinates
    mouse = [0, 0]
    close_req = [False]
    print("Select objects: LEFT-CLICK corners around the object, ENTER or RIGHT-CLICK to close the outline, "
          "BACKSPACE = undo, ESC = clear, C = finish.")

    def on_mouse(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            close_req[0] = True

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    def commit():
        it = poly_item([(x / s, y / s) for x, y in pts], f"object {len(items) + 1}", frame.shape)
        if it is None:
            print("Outline too small - click corners further apart.")
            return False
        items.append(it)
        pts.clear()
        return True

    while True:
        disp = small.copy()
        for it in items:
            _draw_poly(disp, it["poly"], (255, 255, 0), True, s)
            x1, y1 = it["box"][0], it["box"][1]
            cv2.putText(disp, it["name"], (int(x1 * s), max(15, int(y1 * s) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
        _draw_poly(disp, pts, (0, 255, 0), False)
        if pts:
            cv2.line(disp, pts[-1], tuple(mouse), (0, 200, 0), 1)
        msg = (f"Object {len(items) + 1}: click corners ({len(pts)}), ENTER/right-click = close"
               + ("  |  C = done" if items else ""))
        cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(20) & 0xFF
        if close_req[0] or key in (13, 10, 32):          # close polygon
            close_req[0] = False
            if len(pts) >= 3:
                commit()
            elif pts:
                print("Need at least 3 corners.")
        elif key in (8, ord("u")):                         # undo corner
            if pts:
                pts.pop()
        elif key == 27:                                    # clear / finish
            if pts:
                pts.clear()
            elif items:
                break
            else:
                cv2.destroyWindow(WINDOW)
                raise SystemExit("No object selected.")
        elif key in (ord("c"), ord("C")):                  # finish
            if len(pts) >= 3:
                commit()
            if items:
                break
            print("Draw at least one outline first (3+ corners, then ENTER).")
    cv2.destroyWindow(WINDOW)
    return items


def save_items(items, path, quiet=False):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    if not quiet:
        print(f"Saved {len(items)} object(s) to {path}")


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_items(source, path: str, reuse: bool, max_w: int = 1280):
    """Return (reference_frame, items). With reuse=True and a saved file, skip the selection step
    (assumes the camera has not moved and the objects are in place)."""
    if reuse and os.path.exists(path):
        items = load_items(path)
        ok, frame = source.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the video source.")
        print(f"Reusing {len(items)} saved object(s) from {path}")
        return frame, items
    frame = grab_frame(source, max_w)
    items = select_items(frame, max_w)
    save_items(items, path)
    return frame, items
