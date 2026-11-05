"""Follow-the-object mode: keep tracking each object you selected, wherever it goes.

* A visual tracker (OpenCV VitTrack - works for any object, no training) is started on every selected box.
* Each processed frame it gives the object's current box + a confidence score. These boxes are fed to the
  normal pickup logic (src/pickup.py) as if they were detections, so the program knows when somebody picks
  the object up, who is holding it, when it is handed to someone else, and where it was put down
  (that new place becomes its resting place, ready for the next pickup).
* If the tracker loses the object (hidden by a hand or body), the whole frame is searched for it every few
  frames using saved pictures of the object (multi-scale template matching) and the tracker is restarted
  when it is found again.

Soft / deformable / plain objects (a folded shirt, a bag) are hard for any appearance tracker, so three
safeguards are added on top of the tracker:
  1. APPEARANCE CHECK  - a colour model of the object is kept; if the tracked box stops looking like it for a few
     frames (the tracker drifted onto the background) the track is treated as bad even if its score is high.
  2. CARRIED STATE     - if the object is held by a person and the tracker is bad, the object is assumed to be
     travelling with the holder: its box follows the holder's wrist (state "carried"). If the holder leaves the
     view, "left" is reported (the object left with them).
  3. SMARTER RE-SEARCH - while carried, the area around the holder's hands is searched first, using shape (template)
     AND colour (histogram back-projection, which survives folding), then the whole frame with stricter limits.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.detector import Detection
from src.faces import ensure_model

def _ov(a, b):
    """(share of box a inside box b, IoU)"""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa, bb = max(1, (a[2] - a[0]) * (a[3] - a[1])), max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / aa, inter / (aa + bb - inter)


SCALES = (0.5, 0.65, 0.8, 1.0, 1.2, 1.45, 1.75, 2.1, 2.5)   # an object put nearer the camera looks much bigger
BP_ON = 25                                  # back-projection value (0..255) that counts as "object coloured"


def _canon(hsv):
    """Colour noise removal: the hue of a very dark or grey pixel means nothing (a black bottle shows random hues),
    so those pixels are mapped to fixed hue/saturation before any histogram is made or looked up."""
    h = hsv.copy()
    dark = h[..., 2] < 50
    grey = h[..., 1] < 40
    h[..., 0][dark | grey] = 0
    h[..., 1][dark] = 0
    return h


def _color_model(bgr, poly_rel=None):
    """3-D colour histogram (hue, saturation, brightness) of the object, background outside the polygon ignored."""
    mask = None
    if poly_rel and bgr.size:
        h, w = bgr.shape[:2]
        mask = _poly_mask(poly_rel, w, h)
    hsv = _canon(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def _disc_model(hist, bg_bgr):
    """Colour model that favours colours COMMON on the object and RARE in the static scene, so a red chair or a tan
    table is not mistaken for a red/tan object. Falls back to the plain model if nothing is distinctive."""
    bhsv = _canon(cv2.cvtColor(np.clip(bg_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV))
    b = cv2.calcHist([bhsv], [0, 1, 2], None, list(hist.shape), [0, 180, 0, 256, 0, 256])
    b = b / max(float(b.sum()), 1.0)
    h = hist / max(float(hist.sum()), 1e-9)
    r = h / (h + b + 1e-9) * (h > 0.1 * h.max())
    return (r * 255).astype(np.float32) if r.max() > 0.2 else hist


def _backproject(hsv, hist):
    """Per pixel: how typical is this colour for the object (0..255). Manual lookup, because cv2.calcBackProject
    does not handle 3-D histograms on every OpenCV build."""
    hb, sb, vb = hist.shape
    hsv = _canon(hsv)
    idx = (hsv[..., 0].astype(np.int32) * hb // 180, hsv[..., 1].astype(np.int32) * sb // 256,
           hsv[..., 2].astype(np.int32) * vb // 256)
    return hist[idx]


def _fraction_on(bp_crop):
    return float((bp_crop >= BP_ON).mean()) if bp_crop.size else 0.0


class _Engine:
    """Creates trackers. VitTrack (ONNX model, auto-downloaded); falls back to the weaker built-in MIL tracker."""

    def __init__(self, models_dir: str):
        self.net = None
        if hasattr(cv2, "TrackerVit_create"):
            try:
                self.net = ensure_model("vittrack", models_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[follow] VitTrack model unavailable ({exc}); using the simpler MIL tracker.")
        if self.net is None and not hasattr(cv2, "TrackerMIL_create"):
            raise RuntimeError("This OpenCV build has no object tracker. Run: pip install -U opencv-python")
        self.kind = "VitTrack" if self.net else "MIL"

    def new(self, frame, xywh):
        if self.net:
            p = cv2.TrackerVit_Params()
            p.net = self.net
            trk = cv2.TrackerVit_create(p)
        else:
            trk = cv2.TrackerMIL_create()
        trk.init(frame, tuple(int(v) for v in xywh))
        return trk

    def step(self, trk, frame):
        ok, b = trk.update(frame)
        score = float(trk.getTrackingScore()) if self.net else (1.0 if ok else 0.0)
        return bool(ok), [float(v) for v in b], score


def _poly_mask(poly_rel, w, h):
    m = np.zeros((h, w), np.uint8)
    pts = np.array([[u * (w - 1), v * (h - 1)] for u, v in poly_rel], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def _masked(crop, poly_rel):
    """Grey crop with everything outside the polygon replaced by the polygon's mean grey,
    so background pixels do not disturb the re-search."""
    if not poly_rel or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    m = _poly_mask(poly_rel, w, h)
    if m.sum() == 0:
        return crop
    out = crop.copy()
    out[m == 0] = int(crop[m > 0].mean())
    return out


@dataclass
class FollowedObject:
    idx: int
    name: str
    trk: object
    box: tuple[int, int, int, int]
    area0: float
    templates: list                       # [original crop, most recent good crop] (gray)
    score: float = 1.0
    state: str = "tracking"               # tracking | carried | lost
    lost: int = 0
    last_tpl: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=400))
    poly_rel: list | None = None          # polygon outline relative to the box (0..1), None = plain rectangle
    hist: object = None                   # colour model (see _color_model)
    size: tuple = (0, 0)                  # (w, h) of the last good box
    bad: int = 0                          # consecutive frames the tracker looked wrong
    carried: bool = False
    gone: int = 0                         # frames the holder has been out of view while carried
    last_wrist: tuple | None = None
    app: float = 1.0                      # latest appearance score
    size0: tuple = (0, 0)                 # original size (used for search windows; does not shrink)
    cand: tuple | None = None             # candidate box seen while carried (needs confirming)
    cand_n: int = 0
    recent: tuple | None = None           # (person id, frame) of the last hand seen right next to the object
    forced: int | None = None             # person assumed to have taken it (object vanished from its spot)
    dhist: object = None                  # background-aware colour model (see _disc_model)
    rest: tuple | None = None             # where the object sits when nobody touches it
    was_held: bool = False                # trail is only recorded while someone holds the object
    trail_end: int | None = None
    acand: tuple | None = None            # detector candidate waiting for a second agreeing sighting
    acand_n: int = 0
    adrift: int = 0
    moved_n: int = 0                      # frames the object has looked settled at a new place
    fill0: float = 1.0                    # share of its own box that is "object coloured" (a polygon fills less)
    bcand: tuple | None = None            # whole-frame candidate waiting to be confirmed (see _find_anywhere)
    bcand_n: int = 0

    def outline(self):
        """Polygon points (int array) fitted to the current box, or None for rectangles."""
        if not self.poly_rel:
            return None
        x1, y1, x2, y2 = self.box
        return np.array([[x1 + u * (x2 - x1), y1 + v * (y2 - y1)] for u, v in self.poly_rel], np.int32)


class ObjectFollower:
    def __init__(self, cfg: dict, items: list[dict], ref_frame, assist=None):
        self.assist = assist                                         # optional detector helper (src/assist.py)
        self.assist_every = int(cfg.get("assist_every", 15))         # routine check of the tracker against the detector
        self.min_score = float(cfg.get("min_score", 0.35))
        self.reacquire_every = int(cfg.get("reacquire_every", 5))
        self.reacquire_thresh = float(cfg.get("reacquire_thresh", 0.7))
        self.lost_report = int(cfg.get("lost_report_frames", 30))
        self.app_ratio = float(cfg.get("app_ratio", 0.5))           # colour match needed to trust the tracker,
                                                                    # as a share of how much the object fills its box
        self.bad_frames = int(cfg.get("bad_frames", 4))             # frames in a row before a track counts as bad
        self.near_hand_thresh = float(cfg.get("hand_search_thresh", 0.55))   # template match near the holder
        self.hist_thresh_hand = float(cfg.get("hand_color_ratio", 0.7))      # colour match near the holder (x fill)
        self.hist_thresh_full = float(cfg.get("full_color_ratio", 0.9))      # colour match, whole frame (x fill)
        self.leave_frames = int(cfg.get("leave_frames", 25))        # holder out of view this long = "left with them"
        self.recent_frames = int(cfg.get("recent_hand_frames", 20))  # a hand this recent explains a vanished object
        self.gone_diff = float(cfg.get("gone_diff", 22))             # spot looks like empty background if diff < this
        self.anchor_dist = float(cfg.get("anchor_dist", 0.5))         # jump from the resting place (x size) that needs a hand
        self.rest_ncc = float(cfg.get("rest_match", 0.6))              # picture match that proves it is still at rest
        self.carry_every = int(cfg.get("carry_search_every", 2))    # while carried, search near the hands this often
        self.carry_confirm = int(cfg.get("carry_confirm", 3))       # ...and accept only after this many agreeing hits
        self.bg_reject = float(cfg.get("bg_reject", 0.6))           # reject if the empty scene there matched this well (x seen)
        self.bg_alpha = float(cfg.get("bg_alpha", 0.02))
        self.any_every = int(cfg.get("anywhere_every", 3))              # while lost/carried: look in the WHOLE frame this often
        self.any_confirm = int(cfg.get("anywhere_confirm", 3))          # ...and accept after this many agreeing looks (lost)
        self.putdown_confirm = int(cfg.get("putdown_confirm", 4))       # ...and this many while someone still holds it
        self.novelty = float(cfg.get("novelty_diff", 18))                # a new object differs this much from the empty scene
        self.bg = ref_frame.astype(np.float32)                      # slow background picture (people excluded)
        self.eng = _Engine(cfg.get("models_dir", "models"))
        gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        self.objects: list[FollowedObject] = []
        for i, d in enumerate(items):
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            poly_rel = None
            if d.get("poly") and len(d["poly"]) >= 3:
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                poly_rel = [((px - x1) / bw, (py - y1) / bh) for px, py in d["poly"]]
            crop = _masked(gray[y1:y2, x1:x2].copy(), poly_rel)
            trk = self.eng.new(ref_frame, (x1, y1, x2 - x1, y2 - y1))
            o = FollowedObject(i + 1, d["name"], trk, (x1, y1, x2, y2), float((x2 - x1) * (y2 - y1)),
                               [crop, crop.copy()], poly_rel=poly_rel)
            o.hist = _color_model(ref_frame[y1:y2, x1:x2], poly_rel)
            o.size = o.size0 = (x2 - x1, y2 - y1)
            o.rest = (x1, y1, x2, y2)
            self.objects.append(o)
        hole = np.zeros(ref_frame.shape[:2], np.uint8)       # background without the objects (filled from surroundings)
        for o in self.objects:
            x1, y1, x2, y2 = o.box
            cv2.rectangle(hole, (max(0, x1 - 3), max(0, y1 - 3)), (x2 + 3, y2 + 3), 255, -1)
        self.bg = cv2.inpaint(ref_frame, hole, 3, cv2.INPAINT_TELEA).astype(np.float32)
        for o in self.objects:
            x1, y1, x2, y2 = o.box
            o.dhist = _disc_model(o.hist, self.bg)
            o.fill0 = max(0.2, _fraction_on(_backproject(cv2.cvtColor(ref_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV),
                                                         o.dhist)))
        self.just_lost: list[FollowedObject] = []
        self.notes: list[tuple[str, FollowedObject, object]] = []   # ("carried"|"left"|"back", object, person id)
        print(f"Following {len(self.objects)} object(s) with {self.eng.kind}.")

    def name_of(self, idx: int) -> str:
        return next((o.name for o in self.objects if o.idx == idx), f"object {idx}")

    # ------------------------------------------------------------------
    def update(self, frame, frame_idx: int, persons=None, holders=None) -> list[Detection]:
        dets = self._update(frame, frame_idx, persons, holders)
        holders = holders or {}
        for o in self.objects:
            held = holders.get(o.idx) is not None or (o.state == "carried" and o.gone == 0)
            c = ((o.box[0] + o.box[2]) // 2, (o.box[1] + o.box[3]) // 2)
            if held:
                if not o.was_held:                            # pickup: a new path starts at the resting place
                    o.trail.clear()
                    if o.rest is not None:
                        o.trail.append(((o.rest[0] + o.rest[2]) // 2, (o.rest[1] + o.rest[3]) // 2))
                    o.was_held, o.trail_end = True, None
                if o.state != "lost":
                    o.trail.append(c)
            else:
                if o.was_held:                                # put down: path ends here, new resting place
                    o.was_held, o.trail_end = False, frame_idx
                    if o.state != "lost":
                        o.trail.append(c)
                    if o.state == "tracking":
                        o.rest = o.box
                elif o.trail_end is not None and frame_idx - o.trail_end > 75:
                    o.trail.clear()
                    o.trail_end = None
        return dets

    def _update(self, frame, frame_idx: int, persons=None, holders=None) -> list[Detection]:
        """Advance every tracker; return a pseudo-detection (track_id = object number) per found object.
        `persons` / `holders` (person list, {object idx: person id}) enable the carried-state logic."""
        persons = persons or []
        holders = holders or {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = gray.shape
        dets: list[Detection] = []
        self.just_lost = []
        self.notes = []
        self._update_bg(frame, persons)
        if frame_idx % 30 == 0:
            for o in self.objects:
                o.dhist = _disc_model(o.hist, self.bg)
        for o in self.objects:
            holder = holders.get(o.idx)
            if holder is None and o.state == "carried":
                holder = o.forced
            hp = next((p for p in persons if p.track_id == holder), None) if holder is not None else None
            if o.state == "tracking" and holders.get(o.idx) is None:
                near = self._hand_next_to(o, persons)
                if near is not None:
                    o.recent = (near, frame_idx)
            good = False
            if o.state != "carried":
                good, box, score = self._track_ok(o, frame, hsv, W, H)
                if good and self._on_person(o, box, gray, persons):
                    good = False                                   # drifted onto someone's head or clothes
                if good and self.assist is not None:
                    cands = self._assist_boxes(o, frame, frame_idx, holders.get(o.idx) is not None)
                    if cands is not None:
                        bx = box
                        agrees = any(max(_ov(bx, c)) >= 0.25 for c in cands)
                        if cands and not agrees and holders.get(o.idx) is None:
                            o.adrift += 1                          # tracker is where the detector sees nothing like it
                            far = self._assist_pick(o, cands, ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2))
                            if far is not None and o.adrift >= 2:
                                self._restart(o, frame, (far, 0.8))
                                o.adrift, o.acand, o.acand_n, o.bad = 0, None, 0, 0
                                self.notes.append(("reset", o, None))
                                dets.append(Detection(o.idx, o.name, 0.8, o.box))
                                continue
                        elif agrees:
                            o.adrift = 0
                if good and holders.get(o.idx) is None and o.rest is not None:
                    box = self._anchor(o, box, gray, frame, persons, frame_idx)
                if good:
                    o.bad = 0
                    x1, y1, x2, y2 = box
                    o.box, o.score, o.state, o.lost = box, score, "tracking", 0
                    o.size = (x2 - x1, y2 - y1)
                    if score >= 0.6 and o.app >= 0.5 and frame_idx - o.last_tpl >= 20 and not self._overlaps_person(box, persons):   # refresh pictures
                        o.templates[1], o.last_tpl = _masked(gray[y1:y2, x1:x2].copy(), o.poly_rel), frame_idx
                        fresh = _color_model(frame[y1:y2, x1:x2], o.poly_rel)
                        o.hist = 0.8 * o.hist + 0.2 * fresh
                    dets.append(Detection(o.idx, o.name, score, o.box))
                    continue
                o.bad += 1
                if o.bad < self.bad_frames and o.state == "tracking":
                    # brief glitch: keep the last box this frame
                    dets.append(Detection(o.idx, o.name, o.score, o.box))
                    continue
                if hp is None and o.state == "tracking" and o.recent and frame_idx - o.recent[1] <= self.recent_frames \
                        and self._looks_empty(frame, o.box):
                    # a hand was next to it a moment ago and now its spot shows only background: someone took it
                    hp = next((p for p in persons if p.track_id == o.recent[0]), None)
                    if hp is not None and not self._hand_within(o, hp, 3.0):
                        hp = None                                  # that person has walked away: not them
                    if hp is not None:
                        holder = o.forced = o.recent[0]
                if hp is not None and o.state == "tracking":
                    # held by someone and the tracker went bad: assume it travels with the holder
                    o.state, o.carried, o.gone = "carried", True, 0
                    self.notes.append(("carried", o, holder))

            if o.state == "carried":
                if hp is None:                                   # holder not visible
                    if frame_idx % self.any_every == 0:         # ...but the object may be lying somewhere in view
                        fnd = self._find_anywhere(o, frame, gray, hsv, persons, frame_idx, self.any_confirm)
                        if fnd is not None:
                            self._resume(o, frame, fnd, holder, dets)
                            continue
                    o.gone += 1
                    if o.gone >= self.leave_frames:
                        o.state, o.carried, o.lost = "lost", False, self.lost_report
                        self.notes.append(("left", o, holder))
                        continue
                    dets.append(Detection(o.idx, o.name, 0.3, o.box))
                    continue
                o.gone = 0
                found = None
                acands = self._assist_boxes(o, frame, frame_idx, True) if frame_idx % self.carry_every == 0 else None
                if acands:
                    wp = hp.wrists or [((hp.box[0] + hp.box[2]) / 2, (hp.box[1] + hp.box[3]) / 2)]
                    near = [c for c in acands if min(np.hypot((c[0] + c[2]) / 2 - wx, (c[1] + c[3]) / 2 - wy)
                                                      for wx, wy in wp) <= 3.0 * max(o.size0) + 20]
                    pick = self._assist_pick(o, near, wp[0]) if near else None
                    if pick is not None:
                        found = (pick, 0.8)
                if found is None and frame_idx % self.carry_every == 0:
                    cand = self._reacquire_near(gray, hsv, o, hp)
                    if cand is None and frame_idx % (self.carry_every * 6) == 0:
                        cand = self._reacquire(gray, hsv, o)
                    if cand is None:
                        o.cand, o.cand_n = None, 0
                    else:
                        cb = cand[0]
                        agree = (o.cand is not None and
                                 abs((cb[0] + cb[2]) / 2 - (o.cand[0] + o.cand[2]) / 2) < 0.6 * max(o.size0) and
                                 abs((cb[1] + cb[3]) / 2 - (o.cand[1] + o.cand[3]) / 2) < 0.6 * max(o.size0))
                        o.cand, o.cand_n = cb, (o.cand_n + 1 if agree else 1)
                        if o.cand_n >= self.carry_confirm:
                            found = cand
                if found is None and frame_idx % (self.any_every * 2) == 0:
                    # put down somewhere away from every hand? (the holder may still be standing right there)
                    wr = [w for q in persons for w in (q.wrists or [])]
                    found = self._find_anywhere(o, frame, gray, hsv, persons, frame_idx, self.putdown_confirm, away=wr)
                if found is not None:
                    self._resume(o, frame, found, holder, dets)
                    continue
                o.box = self._box_at_wrist(o, hp, W, H)
                dets.append(Detection(o.idx, o.name, 0.5, o.box))
                continue

            # --- not held and not found: plain "lost"; first look at its resting place, then the whole frame
            o.state, o.lost = "lost", o.lost + 1
            if o.lost == self.lost_report:
                self.just_lost.append(o)
            if o.lost % self.any_every == 0:
                fnd = self._find_anywhere(o, frame, gray, hsv, persons, frame_idx, self.any_confirm)
                if fnd is not None:
                    self._resume(o, frame, fnd, None, dets)
                    continue
            if o.lost % self.reacquire_every == 0:
                found = None
                acands = self._assist_boxes(o, frame, frame_idx, False)
                if acands:
                    pick = self._assist_pick(o, [c for c in acands if not self._overlaps_person(c, persons)] or acands,
                                             ((o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2))
                    if pick is not None:
                        found = (pick, 0.8)
                elif acands is not None:
                    o.acand, o.acand_n = None, 0
                at = self._object_at(o, gray, o.rest) if (found is None and o.rest is not None) else None
                if found is None and at is not None and self._hand_near_box(o.rest, persons, 1.5) is None:
                    found = (at, 0.8)
                if found is None:
                    cand = self._reacquire(gray, hsv, o)
                    if cand is None:
                        o.cand, o.cand_n = None, 0
                    else:
                        cb = cand[0]
                        agree = (o.cand is not None and
                                 abs((cb[0] + cb[2]) / 2 - (o.cand[0] + o.cand[2]) / 2) < 0.6 * max(o.size0) and
                                 abs((cb[1] + cb[3]) / 2 - (o.cand[1] + o.cand[3]) / 2) < 0.6 * max(o.size0))
                        o.cand, o.cand_n = cb, (o.cand_n + 1 if agree else 1)
                        if o.cand_n >= 2 and self._overlaps_person(cb, persons) is False:
                            found = cand
                if found is not None:
                    o.cand, o.cand_n = None, 0
                    self._restart(o, frame, found)
                    o.state, o.bad = "tracking", 0
                    o.rest = o.box
                    dets.append(Detection(o.idx, o.name, found[1], o.box))
        return dets

    # ------------------------------------------------------------------
    # Continuous tracking: find the object ANYWHERE in the frame (it may have been put down at a new place)
    def _resume(self, o, frame, found, holder, dets):
        """The object was found again: restart its tracker there; that place is its new resting place."""
        o.cand, o.cand_n, o.bcand, o.bcand_n = None, 0, None, 0
        self._restart(o, frame, found)
        o.state, o.carried, o.bad, o.forced, o.gone, o.lost = "tracking", False, 0, None, 0, 0
        o.rest = o.box
        self.notes.append(("back", o, holder))
        dets.append(Detection(o.idx, o.name, found[1], o.box))

    def _diff_map(self, frame, persons):
        """(half-size difference picture between now and the empty scene, scale). Brightness changes of the whole
        camera picture are compensated, people are blanked out."""
        H, W = frame.shape[:2]
        f = 0.5
        small = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float32)
        bg = cv2.resize(self.bg, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        valid = np.full(small.shape[:2], 255, np.uint8)
        for p in persons:
            x1, y1, x2, y2 = p.box
            mx, my = int(0.12 * (x2 - x1)), int(0.08 * (y2 - y1))
            cv2.rectangle(valid, (int((x1 - mx) * f), int((y1 - my) * f)), (int((x2 + mx) * f), int((y2 + my) * f)), 0, -1)
        m = valid[::3, ::3] > 0
        gain = np.ones(3, np.float32)
        if m.sum() > 500:
            for c in range(3):
                den = float(np.median(small[::3, ::3, c][m]))
                gain[c] = float(np.clip(float(np.median(bg[::3, ::3, c][m])) / max(den, 1.0), 0.6, 1.6))
        diff = np.abs(small * gain - bg).mean(axis=2)
        diff[valid == 0] = 0
        return diff, f

    def _blobs(self, frame, persons):
        """Boxes of things that are in the picture now but not in the empty scene (candidates for a put-down object)."""
        diff, f = self._dm
        fg = (diff > max(22.0, self.novelty + 6)).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            out.append((int(x / f), int(y / f), int((x + w) / f), int((y + h) / f)))
        return out

    def _novel(self, frame, persons, box):
        """True if this box looks different from the empty scene (so it is a new thing, not a fixed one)."""
        diff, f = self._dm
        x1, y1, x2, y2 = (int(v * f) for v in box)
        patch = diff[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        return patch.size > 0 and float(patch.mean()) >= self.novelty

    def _cand_score(self, o, frame, hsv, box):
        """How much does `box` look like this object (size, shape, colours)? 0 = not at all."""
        H, W = hsv.shape[:2]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            return 0.0, None
        d0 = float(np.hypot(*o.size0))
        if not 0.4 * d0 <= float(np.hypot(w, h)) <= 2.8 * d0:
            return 0.0, None
        a0 = max(o.size0[0], 1) / max(o.size0[1], 1)
        a = w / h
        r_up, r_lie = a / a0, a * a0                                   # same shape, standing or lying
        if not (0.4 <= r_up <= 2.5 or 0.4 <= r_lie <= 2.5):
            return 0.0, None
        if w * h < 0.3 * o.size0[0] * o.size0[1]:                       # far too small an area
            return 0.0, None
        shape = max(min(r_up, 1 / r_up), 0.7 * min(r_lie, 1 / r_lie))
        size = min(float(np.hypot(w, h)) / d0, d0 / float(np.hypot(w, h)))
        bp = _backproject(hsv[y1:y2, x1:x2], self._m(o))
        on = (bp >= BP_ON)
        frac = float(on.mean())
        if frac < max(0.15, 0.5 * o.fill0):
            return 0.0, None
        ys, xs = np.where(on)                                           # tighten the box around the object-coloured part
        if len(xs) >= 20:
            tx1, tx2 = np.percentile(xs, [3, 97]); ty1, ty2 = np.percentile(ys, [3, 97])
            px, py = 0.06 * (tx2 - tx1), 0.04 * (ty2 - ty1)
            nb = (int(x1 + tx1 - px), int(y1 + ty1 - py), int(x1 + tx2 + px), int(y1 + ty2 + py))
            nb = (max(0, nb[0]), max(0, nb[1]), min(W, nb[2]), min(H, nb[3]))
            if nb[2] - nb[0] >= 8 and nb[3] - nb[1] >= 8:
                box = nb
        else:
            box = (x1, y1, x2, y2)
        return frac * shape * size, box

    def _find_anywhere(self, o, frame, gray, hsv, persons, frame_idx, need, away=None):
        """Look for the object in the WHOLE frame: detector boxes of its kind plus everything new in the picture
        (not in the empty scene). The best match must stay in the same place for `need` looks in a row, so a passing
        shadow or a person's clothes is not mistaken for it. `away`: points (hands) the object must be far from."""
        self._dm = self._diff_map(frame, persons)
        cands = []
        ab = self._assist_boxes(o, frame, frame_idx, False)
        if ab:
            cands += [(b, 1.0) for b in ab]
        cands += [(b, 0.9) for b in self._blobs(frame, persons)]
        best, bs = None, 0.0
        reach = 1.5 * max(o.size0) + 10
        for b, w in cands:
            if self._overlaps_person(b, persons) or not self._novel(frame, persons, b):
                continue
            if away and any(max(b[0] - wx, 0, wx - b[2]) ** 2 + max(b[1] - wy, 0, wy - b[3]) ** 2 < reach * reach
                            for wx, wy in away):
                continue
            sc, nb = self._cand_score(o, frame, hsv, b)
            if nb is not None and sc * w > bs:
                best, bs = nb, sc * w
        if best is None:
            o.bcand, o.bcand_n = None, 0
            return None
        agree = (o.bcand is not None and
                 abs((best[0] + best[2]) / 2 - (o.bcand[0] + o.bcand[2]) / 2) < 0.6 * max(o.size0) and
                 abs((best[1] + best[3]) / 2 - (o.bcand[1] + o.bcand[3]) / 2) < 0.6 * max(o.size0))
        o.bcand, o.bcand_n = best, (o.bcand_n + 1 if agree else 1)
        return (best, min(0.95, 0.5 + bs)) if o.bcand_n >= need else None

    # ------------------------------------------------------------------
    def _track_ok(self, o, frame, hsv, W, H):
        """Run the tracker one step; good = it found something sane AND the box still looks like the object."""
        ok, (x, y, w, h), score = self.eng.step(o.trk, frame)
        cx, cy = x + w / 2, y + h / 2
        sane = (w >= 8 and h >= 8 and o.area0 * 0.2 <= w * h <= o.area0 * 5 and 0 <= cx < W and 0 <= cy < H)
        if not (ok and score >= self.min_score and sane):
            return False, None, score
        x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return False, None, score
        bp = _backproject(hsv[y1:y2, x1:x2], self._m(o))
        o.app = _fraction_on(bp)
        if o.app < max(0.12, self.app_ratio * o.fill0):   # confident but looks wrong = drifted onto the background
            return False, None, score
        return True, (x1, y1, x2, y2), score

    def _assist_boxes(self, o, frame, frame_idx, holder_present):
        """Detector candidates for this object that also look like it (colours, size). Runs the detector only when
        useful: object lost/carried/held, tracker unsure, or on a slow routine check."""
        a = self.assist
        if a is None or not a.wants(o.idx):
            return None
        busy = o.state != "tracking" or holder_present or o.bad > 0 or frame_idx % self.assist_every == 0
        if not busy:
            return None
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = []
        d0 = float(np.hypot(*o.size0))
        for b in a.boxes(frame, frame_idx, o.idx):
            x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            d = float(np.hypot(x2 - x1, y2 - y1))
            if not 0.45 * d0 <= d <= 2.8 * d0:
                continue
            if _fraction_on(_backproject(hsv[y1:y2, x1:x2], o.hist)) < 0.3:   # wrong colours: another bottle/phone
                continue
            out.append((x1, y1, x2, y2))
        return out

    def _assist_pick(self, o, cands, ref):
        """Candidate closest to `ref` (x, y), confirmed by two sightings in the same place; None until then."""
        if not cands:
            o.acand, o.acand_n = None, 0
            return None
        cb = min(cands, key=lambda b: (((b[0] + b[2]) / 2 - ref[0]) ** 2 + ((b[1] + b[3]) / 2 - ref[1]) ** 2))
        agree = (o.acand is not None and
                 abs((cb[0] + cb[2]) / 2 - (o.acand[0] + o.acand[2]) / 2) < 0.8 * max(o.size0) and
                 abs((cb[1] + cb[3]) / 2 - (o.acand[1] + o.acand[3]) / 2) < 0.8 * max(o.size0))
        o.acand, o.acand_n = cb, (o.acand_n + 1 if agree else 1)
        return cb if o.acand_n >= 2 else None

    @staticmethod
    def _overlaps_person(box, persons):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return any(p.box[0] <= cx <= p.box[2] and p.box[1] <= cy <= p.box[3] for p in persons)

    def _on_person(self, o, box, gray, persons):
        """True if the tracked box sits on a person's head or body with no hand next to it (dark hair, dark clothes
        look like a dark object). Not applied when the object's own picture is still visible at its resting place."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for p in persons:
            x1, y1, x2, y2 = p.box
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            head = cy < y1 + 0.22 * (y2 - y1)
            if not head and self._hand_near_box(box, [p], 1.2) is not None:
                continue
            if o.rest is not None and not head and self._object_at(o, gray, box) is not None \
                    and np.hypot(cx - (o.rest[0] + o.rest[2]) / 2, cy - (o.rest[1] + o.rest[3]) / 2) < 0.5 * max(o.size0):
                continue
            return True
        return False

    def _anchor(self, o, box, gray, frame, persons, frame_idx):
        """Rest anchoring: with nobody's hand near its resting place an object cannot move. If the tracker says it
        jumped away while the object is still visibly there, the tracker is put back on the real object."""
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = o.rest
        d = np.hypot((x1 + x2) / 2 - (rx1 + rx2) / 2, (y1 + y2) / 2 - (ry1 + ry2) / 2)
        far = d > self.anchor_dist * max(o.size0)
        if self._hand_near_box(o.rest, persons, 1.5) is not None:
            o.moved_n = 0
            return box                                     # someone is touching it: it may move
        at = self._object_at(o, gray, o.rest)
        if far and at is not None:                         # still there, tracker jumped: snap back
            self._restart(o, frame, (at, 0.9))
            o.bad, o.moved_n = 0, 0
            self.notes.append(("reset", o, None))
            return o.box
        if far:                                            # tracker elsewhere, nobody touched it: keep the old place
            return box
        else:                                              # settled: follow tiny shifts (light, vibration)
            o.rest = tuple(int(0.9 * a + 0.1 * b) for a, b in zip(o.rest, box))
            o.moved_n = 0
        return box

    def _object_at(self, o, gray, box):
        """Box where the object's original picture matches near `box`, or None (it is not there)."""
        H, W = gray.shape
        x1, y1, x2, y2 = box
        m = 12
        rx1, ry1, rx2, ry2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
        roi = gray[ry1:ry2, rx1:rx2]
        best, loc, bt = 0.0, None, None
        for t in o.templates:
            th, tw = t.shape[:2]
            if th < 8 or tw < 8 or th > roi.shape[0] or tw > roi.shape[1]:
                continue
            res = np.nan_to_num(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx > best:
                best, loc, bt = float(mx), ml, t
        if loc is None or best < self.rest_ncc:
            return None
        th, tw = bt.shape[:2]
        return (rx1 + loc[0], ry1 + loc[1], rx1 + loc[0] + tw, ry1 + loc[1] + th)

    @staticmethod
    def _hand_near_box(box, persons, mult):
        x1, y1, x2, y2 = box
        m = mult * max(x2 - x1, y2 - y1)
        for p in persons:
            if p.track_id is None:
                continue
            for wx, wy in p.wrists:
                if max(x1 - wx, 0, wx - x2) ** 2 + max(y1 - wy, 0, wy - y2) ** 2 <= m * m:
                    return p.track_id
        return None

    @staticmethod
    def _m(o):
        return o.dhist if o.dhist is not None else o.hist

    def _hand_next_to(self, o, persons):
        """Id of a person whose wrist is right next to the object's box (None if nobody)."""
        x1, y1, x2, y2 = o.box
        m = 1.2 * max(x2 - x1, y2 - y1)
        for p in persons:
            if p.track_id is None:
                continue
            for wx, wy in p.wrists:
                if max(x1 - wx, 0, wx - x2) ** 2 + max(y1 - wy, 0, wy - y2) ** 2 <= m * m:
                    return p.track_id
        return None

    @staticmethod
    def _hand_within(o, person, mult):
        """True if one of the person's wrists is within `mult` x the object's size of its box."""
        x1, y1, x2, y2 = o.box
        m = mult * max(x2 - x1, y2 - y1)
        pts = person.wrists or [((person.box[0] + person.box[2]) / 2, (person.box[1] + person.box[3]) / 2)]
        return any(max(x1 - wx, 0, wx - x2) ** 2 + max(y1 - wy, 0, wy - y2) ** 2 <= m * m for wx, wy in pts)

    def _looks_empty(self, frame, box):
        """True if this spot looks like the empty background picture (the object is simply not there)."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return False
        diff = np.abs(frame[y1:y2, x1:x2].astype(np.float32) - self.bg[y1:y2, x1:x2]).mean()
        return float(diff) < self.gone_diff

    def _restart(self, o, frame, found):
        (fx1, fy1, fx2, fy2), sim = found
        o.trk = self.eng.new(frame, (fx1, fy1, fx2 - fx1, fy2 - fy1))
        o.box, o.score, o.lost = (fx1, fy1, fx2, fy2), sim, 0
        o.size = (fx2 - fx1, fy2 - fy1)

    def _box_at_wrist(self, o, hp, W, H):
        """Box of the object's last size, centred on the holder's wrist nearest to where it was."""
        w, h = o.size
        ref = o.last_wrist or ((o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2)
        if hp.wrists:
            wx, wy = min(hp.wrists, key=lambda q: (q[0] - ref[0]) ** 2 + (q[1] - ref[1]) ** 2)
        else:
            x1, y1, x2, y2 = hp.box
            wx, wy = (x1 + x2) / 2, y1 + 0.6 * (y2 - y1)
        o.last_wrist = (wx, wy)
        x1, y1 = int(min(max(wx - w / 2, 0), W - w)), int(min(max(wy - h / 2, 0), H - h))
        return (x1, y1, x1 + int(w), y1 + int(h))

    # ------------------------------------------------------------------
    def _search(self, gray, hsv, o, region, thr_tmpl, thr_hist):
        """Look for the object inside `region` (x1,y1,x2,y2): template match first (shape), then colour."""
        rx1, ry1, rx2, ry2 = region
        g, hs = gray[ry1:ry2, rx1:rx2], hsv[ry1:ry2, rx1:rx2]
        if g.shape[0] < 12 or g.shape[1] < 12:
            return None
        big = max(g.shape) > 500
        f = 0.5 if big else 1.0
        small = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f != 1.0 else g
        best, best_box = 0.0, None
        for tmpl in o.templates:
            for sc in SCALES:
                t = cv2.resize(tmpl, None, fx=f * sc, fy=f * sc, interpolation=cv2.INTER_AREA)
                th, tw = t.shape[:2]
                if th < 12 or tw < 12 or th >= small.shape[0] or tw >= small.shape[1]:
                    continue
                res = np.nan_to_num(cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED))
                _, mx, _, loc = cv2.minMaxLoc(res)
                if mx > best:
                    best = float(mx)
                    best_box = (int(loc[0] / f), int(loc[1] / f), int((loc[0] + tw) / f), int((loc[1] + th) / f))
        if best_box is not None and best >= thr_tmpl:
            b = best_box
            tbox = (rx1 + b[0], ry1 + b[1], rx1 + b[2], ry1 + b[3])
            seen = _fraction_on(_backproject(hs[b[1]:b[3], b[0]:b[2]], self._m(o)))       # colours right?
            if seen >= max(0.12, self.app_ratio * o.fill0) and not self._colour_in_background(o, tbox, seen):
                return tbox, best
        # colour search: window of the object's size with the highest share of object-coloured pixels
        bp = _backproject(hs, self._m(o))
        on = (bp >= BP_ON).astype(np.float32)
        bw, bh = o.size0
        cand, cscore = None, 0.0
        for sc in (0.7, 1.0, 1.4, 2.0):
            ww, hh = int(bw * sc), int(bh * sc)
            if ww < 8 or hh < 8 or ww >= on.shape[1] or hh >= on.shape[0]:
                continue
            m = cv2.boxFilter(on, -1, (ww, hh), normalize=True, borderType=cv2.BORDER_CONSTANT)
            half = (ww // 2, hh // 2)
            m[:half[1], :] = 0; m[-half[1]:, :] = 0; m[:, :half[0]] = 0; m[:, -half[0]:] = 0
            _, mx, _, loc = cv2.minMaxLoc(m)
            if mx > cscore:
                cscore, cand = float(mx), (loc[0] - ww // 2, loc[1] - hh // 2, loc[0] + ww - ww // 2, loc[1] + hh - hh // 2)
        if cand is not None and cscore >= max(0.2, thr_hist * o.fill0):
            box = (rx1 + cand[0], ry1 + cand[1], rx1 + cand[2], ry1 + cand[3])
            if not self._colour_in_background(o, box, cscore):
                return box, cscore
        return None

    def _update_bg(self, frame, persons):
        """Slow-moving picture of the empty scene: people (and a margin) are left out of the update."""
        mask = np.full(frame.shape[:2], 255, np.uint8)
        for p in persons:
            x1, y1, x2, y2 = p.box
            mx, my = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
            cv2.rectangle(mask, (max(0, x1 - mx), max(0, y1 - my)), (x2 + mx, y2 + my), 0, -1)
        for o in self.objects:                           # the objects themselves never become "background"
            if o.state != "lost":
                x1, y1, x2, y2 = o.box
                cv2.rectangle(mask, (max(0, x1 - 6), max(0, y1 - 6)), (x2 + 6, y2 + 6), 0, -1)
        for o in self.objects:                           # where an object is supposed to rest: never "background"
            if o.rest is not None:
                x1, y1, x2, y2 = o.rest
                cv2.rectangle(mask, (max(0, x1 - 8), max(0, y1 - 8)), (x2 + 8, y2 + 8), 0, -1)
        if any(o.state in ("lost", "carried") for o in self.objects):
            return            # while an object is missing or in someone's hand, freeze the empty-scene picture:
                              # otherwise the object, once put down somewhere, soaks into the "background"
        cv2.accumulateWeighted(frame, self.bg, self.bg_alpha, mask=mask)

    def _colour_in_background(self, o, box, seen):
        """True if that spot already looked about as object-coloured in the static scene before (so the candidate is
        probably just background): compares the empty-scene picture with what is seen there now."""
        H, W = self.bg.shape[:2]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return True
        patch = cv2.cvtColor(np.clip(self.bg[y1:y2, x1:x2], 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        return _fraction_on(_backproject(patch, self._m(o))) >= max(0.08, self.bg_reject * seen)

    def _reacquire_near(self, gray, hsv, o, hp):
        """Search around the holder's hands (or body) first - the object is almost certainly there."""
        H, W = gray.shape
        w, h = o.size0
        r = int(2.5 * max(w, h)) + 20
        pts = list(hp.wrists) or [((hp.box[0] + hp.box[2]) / 2, (hp.box[1] + hp.box[3]) / 2)]
        best = None
        for (px, py) in pts:
            region = (int(max(0, px - r)), int(max(0, py - r)), int(min(W, px + r)), int(min(H, py + r)))
            res = self._search(gray, hsv, o, region, self.near_hand_thresh, self.hist_thresh_hand)
            if res is not None and (best is None or res[1] > best[1]):
                best = res
        return self._clip(best, W, H)

    def _reacquire(self, gray, hsv, o):
        """Whole-frame search with stricter limits (more chance of a false match)."""
        H, W = gray.shape
        return self._clip(self._search(gray, hsv, o, (0, 0, W, H), self.reacquire_thresh, self.hist_thresh_full), W, H)

    @staticmethod
    def _clip(found, W, H):
        if found is None:
            return None
        (x1, y1, x2, y2), sc = found
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        return ((x1, y1, x2, y2), sc) if x2 - x1 >= 8 and y2 - y1 >= 8 else None
