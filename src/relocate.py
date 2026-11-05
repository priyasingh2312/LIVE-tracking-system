"""Finds where a saved object really is now, for "use the same objects as last time".

The saved outline only says where the object WAS. Before tracking starts, each saved object's original picture
(data/objects/<id>/crop.jpg) is searched for in the WHOLE frame (several sizes, upright and turned on its side) and the outline
is moved to where it is found. If the object is still in its saved place nothing moves."""
from __future__ import annotations

import os

import cv2
import numpy as np

SCALES = tuple(round(float(v), 3) for v in np.arange(0.35, 1.71, 0.04))


def _mask(shape, poly, box):
    h, w = shape[:2]
    m = np.full((h, w), 255, np.uint8)
    if poly and len(poly) >= 3:
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pts = np.array([[(px - x1) / bw * w, (py - y1) / bh * h] for px, py in poly], np.int32)
        m[:] = 0
        cv2.fillPoly(m, [pts], 255)
    return m


def _auto_mask(crop):
    """No outline shape was saved (a plain rectangle): cut the object out of its box with GrabCut so the background is ignored."""
    h, w = crop.shape[:2]
    full = np.full((h, w), 255, np.uint8)
    try:
        m = np.zeros((h, w), np.uint8)
        rect = (max(1, int(w * .08)), max(1, int(h * .05)), max(2, int(w * .84)), max(2, int(h * .9)))
        cv2.grabCut(crop, m, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
        out = np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8)
        return out if 0.25 < out.mean() / 255 < 0.95 else full
    except cv2.error:
        return full


def _hs(bgr, mask=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask, [12, 6], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _score_at(gray, bgr, tmpl, tmask, tcol, x, y, tbgr):
    th, tw = tmpl.shape[:2]
    if x < 0 or y < 0 or y + th > gray.shape[0] or x + tw > gray.shape[1]:
        return -1.0
    patch = bgr[y:y + th, x:x + tw]
    sel = tmask > 0
    if sel.sum() < 20:
        return -1.0
    dm = float(np.abs(patch[sel].mean(0) - tbgr[sel].mean(0)).max())          # average colour must be close
    if dm > 45:
        return -1.0
    if gray[y:y + th, x:x + tw][sel].std() < 0.4 * max(tmpl[sel].std(), 1.0):  # a flat patch (blank wall) is not the object
        return -1.0
    r = cv2.matchTemplate(gray[y:y + th, x:x + tw], tmpl, cv2.TM_CCOEFF_NORMED, mask=tmask)
    ncc = float(np.nan_to_num(r[0, 0], nan=-1, posinf=-1, neginf=-1))
    col = float(cv2.compareHist(tcol, _hs(patch, tmask), cv2.HISTCMP_CORREL))
    return 0.6 * ncc + 0.4 * max(col, 0.0)


def _canon_bins(bgr):
    """Per-pixel colour bin: hue only counts for coloured pixels (dark, grey and washed-out pixels share one bin each)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, sa, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vb = np.minimum(v // 64, 3)
    plain = (sa < 50) | (v < 50)
    hb = np.where(plain, 12, np.minimum(h // 15, 11))
    sb = np.where(plain, 0, np.minimum(sa // 86, 2))
    return (hb * 3 + sb) * 4 + vb                                   # 13*3*4 bins


def _find_by_colour(frame, crop, mask, accept=0.55):
    """Colour + shape search: where does a blob of the object's colours, with the object's proportions, sit in the frame?"""
    nb = 13 * 3 * 4
    cb = _canon_bins(crop)
    hist = np.bincount(cb[mask > 0], minlength=nb).astype(np.float32)
    if hist.sum() < 20:
        return None
    hist /= hist.sum()
    bg = np.bincount(_canon_bins(crop)[mask == 0], minlength=nb).astype(np.float32)
    if bg.sum() > 0:
        hist = np.maximum(hist - 0.5 * bg / bg.sum(), 0)                 # colours the object shares with its old surroundings count less
    if hist.max() <= 0:
        return None
    lut = hist / hist.max()
    bp = (lut[_canon_bins(frame)] > 0.25).astype(np.uint8) * 255
    bp = cv2.morphologyEx(bp, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))      # splits thin links to dark patterns nearby
    n, lab, st, _ = cv2.connectedComponentsWithStats(bp, 8)
    ch, cw = crop.shape[:2]
    ys, xs = np.where(mask > 0)
    oh, ow = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1            # the object's own proportions
    asp = oh / ow
    fill0 = float((mask > 0).sum()) / (oh * ow)
    best = (-1.0, None)
    fh, fw = frame.shape[:2]
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 40 or w < 6 or h < 6:
            continue
        cands = [(int(x), int(y), int(w), int(h), int(a))]
        comp = (lab[y:y + h, x:x + w] == i)
        wid = comp.sum(1)                                              # width per row: trim a shadow / dark patch hanging off the object
        top = np.median(wid[:max(3, int(h * 0.6))])
        keep, bad = 0, 0
        for r_ in range(h):                                            # short wide bumps (a shadow, a dark spot) do not end the object
            if wid[r_] <= 1.35 * max(top, 1) + 2:
                keep, bad = r_ + 1, 0
            else:
                bad += 1
                if bad > 10:
                    break
        if 6 <= keep < h or True:
            sub = comp[:keep]
            cols = np.where(sub.sum(0) >= 0.3 * keep)[0]                  # the solid columns, not stray side bits
            if cols.size == 0:
                continue
            cands.append((int(x + cols.min()), int(y), int(cols.max() - cols.min() + 1), int(keep), int(sub.sum())))
        for cx, cy, cw_, ch_, ca in cands:
            if cx <= 1 or cy <= 1 or cx + cw_ >= fw - 1 or cy + ch_ >= fh - 1:
                continue                                               # cut by the picture edge: not a whole object
            scale = float(np.sqrt(cw_ * ch_ / float(ow * oh)))
            if not 0.4 < scale < 1.8:
                continue                                               # far too small or too big to be the same object
            if not 0.6 < (ch_ / cw_) / asp < 1.7:
                continue                                               # wrong proportions (objects lying down are not searched)
            a_sim = 1 - min(1.0, abs(np.log((ch_ / cw_) / asp)))
            f_sim = 1 - min(1.0, abs(ca / (cw_ * ch_) - fill0) / max(fill0, 0.3))
            sc = 0.55 * a_sim + 0.45 * f_sim
            if sc > best[0]:
                best = (sc, (cx, cy, cx + cw_, cy + ch_))
    if best[1] is None or best[0] < accept:
        return None
    return best[1], best[0]


def find_saved(frame, it: dict, data_dir: str, accept: float = 0.5):
    """Return the new (x1, y1, x2, y2) box and score for one saved item, or None if the picture is missing / not found."""
    oid = it.get("oid")
    crop = cv2.imread(os.path.join(data_dir, "objects", str(oid), "crop.jpg")) if oid else None
    if crop is None or crop.size == 0:
        return None
    box = tuple(int(v) for v in it["box"])
    has_poly = bool(it.get("poly") and len(it["poly"]) >= 3)
    mask0 = _mask(crop.shape, it.get("poly"), box) if has_poly else _auto_mask(crop)
    cres = _find_by_colour(frame, crop, mask0)
    if cres is not None:
        return cres
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = (-1.0, None)
    for turn in (0, 1):
        c, m = (crop, mask0) if turn == 0 else (cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask0, cv2.ROTATE_90_CLOCKWISE))
        for s in SCALES:
            t = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tm = cv2.resize(m, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
            if t.shape[0] < 8 or t.shape[1] < 8 or t.shape[0] >= frame.shape[0] or t.shape[1] >= frame.shape[1]:
                continue
            tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            if tg.std() < 3:
                continue
            res = cv2.matchTemplate(gray, tg, cv2.TM_CCOEFF_NORMED, mask=tm)
            res = np.nan_to_num(res, nan=-1, posinf=-1, neginf=-1)
            tcol = _hs(t, tm)
            for _ in range(3):                                   # best few peaks at this size, checked with colour too
                _, mx, _, loc = cv2.minMaxLoc(res)
                if mx < 0.2:
                    break
                x, y = loc
                sc = _score_at(gray, frame, tg, tm, tcol, x, y, t)
                if sc > best[0]:
                    best = (sc, (x, y, x + t.shape[1], y + t.shape[0]))
                cv2.rectangle(res, (max(0, x - t.shape[1] // 2), max(0, y - t.shape[0] // 2)),
                              (x + t.shape[1] // 2, y + t.shape[0] // 2), -1, -1)
    if not has_poly and mask0.min() == 255:
        accept = max(accept, 0.72)                       # could not cut the object out: the background is in the picture, be strict
    if best[1] is None or best[0] < accept:
        return None
    return best[1], best[0]


def relocate_items(frame, items: list, data_dir: str, keep: float = 0.72) -> list:
    """Move each saved item to where it is now. Sets it['box'] (and drops the old outline, which no longer fits)."""
    for it in items:
        name = it.get("name", "object")
        old = tuple(int(v) for v in it["box"])
        found = find_saved(frame, it, data_dir)
        if found is None:
            print(f"{name}: could not be found in the picture. It is still tracked at its saved place; "
                  f"restart without 'Use the same objects' to outline it again.")
            continue
        box, sc = found
        ox, oy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
        nx, ny = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if abs(ox - nx) + abs(oy - ny) > 0.6 * max(old[2] - old[0], old[3] - old[1]):
            print(f"{name}: found at a new place (match {sc:.2f}); the outline was moved there.")
            it["box"] = list(box)
            it.pop("poly", None)
        else:
            print(f"{name}: still at its saved place (match {sc:.2f}).")
    return items
