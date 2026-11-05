"""Who took the object, and are they allowed to?

For every tracked person we keep a small buffer of their best face observations (cached over time,
so a face seen BEFORE the pickup still counts). When an object is taken we decide:
  * AUTHORISED     - the person matches an enrolled (known) person;
  * UNAUTHORISED   - a usable face was seen but matches nobody: alert, save the face, remember it as
                     "unknown #N" so the same person is recognised (and counted) next time;
  * UNIDENTIFIED   - no usable face was captured within `face_wait_sec` AND no usable body picture either: alert
                     (fail-safe, never a silent pass).

Body fingerprints (src/body.py) are used ONLY to recognise a stranger again when the face is hidden - never to
authorise anybody. A stranger's face and body are stored together as one "unknown #N".
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.facedb import FaceDB
from src.faces import FaceEngine, FaceObs


@dataclass
class Decision:
    status: str                 # authorised | unauthorised | unidentified
    item: str
    person_id: int | None
    name: str | None = None
    unknown_id: int | None = None
    times_seen: int = 0
    sim: float = 0.0
    snapshot: str = ""
    alert: bool = False         # False for authorised, or for a duplicate alert that was suppressed
    note: str = ""
    by_body: bool = False       # the stranger was recognised / stored by body because no face was visible
    body_sim: float = 0.0


@dataclass
class _Track:
    last_seen: float
    last_try: int = -10_000
    obs: list = field(default_factory=list)       # best FaceObs, sorted by quality desc
    name: str | None = None                       # cached identity once confirmed
    hits: dict = field(default_factory=dict)
    sim: float = 0.0
    body_obs: list = field(default_factory=list)  # best BodyObs, sorted by quality desc
    last_body: int = -10_000
    last_dbg: float = 0.0
    sims: list = field(default_factory=list)
    unknown_id: int | None = None                 # number of the saved stranger this person was matched to


@dataclass
class _Pending:
    item: str
    person_id: int | None
    person_box: tuple | None
    frame: np.ndarray
    ts: float
    deadline: float


class Authorizer:
    def __init__(self, cfg: dict, db: FaceDB, engine: FaceEngine, body=None):
        self.db, self.engine, self.body = db, engine, body
        self.body_thr = float(cfg.get("body_thresh", 0.75))      # cosine similarity to link two body pictures
        self.body_every = int(cfg.get("body_every", 20))         # body picture of each person every N frames
        self.max_body = int(cfg.get("max_body_obs", 4))
        self.data_dir = cfg.get("data_dir", "data")
        self.known_thr = float(cfg.get("known_thresh", 0.45))
        self.confirm_hits = int(cfg.get("known_hits", 3))          # good matches needed before someone is named
        self.debug = bool(cfg.get("debug_faces", False))           # print every face similarity (for tuning)
        self.unknown_thr = float(cfg.get("unknown_thresh", 0.363))
        self.face_every = int(cfg.get("face_every", 4))
        self.min_person_w = int(cfg.get("min_person_width", 100))
        self.q_good = float(cfg.get("q_good", 0.5))
        self.min_obs = int(cfg.get("face_min_obs", 4))             # good face samples needed before calling someone a stranger
        self.wait = float(cfg.get("face_wait_sec", 5.0))
        self.max_obs = int(cfg.get("max_obs", 6))
        self.dedupe = float(cfg.get("alert_dedupe_sec", 60))
        self.beep = bool(cfg.get("beep", True))
        self.tracks: dict[int, _Track] = {}
        self.pending: list[_Pending] = []
        self._last_alert: dict[tuple, float] = {}

    # ---- every processed frame --------------------------------------------------
    def observe(self, frame, persons, now: float, frame_idx: int):
        busy = {p.person_id for p in self.pending}
        body_budget = 1                                   # at most one body picture per processed frame (CPU)
        for p in persons:
            if p.track_id is None:
                continue
            t = self.tracks.setdefault(p.track_id, _Track(last_seen=now))
            t.last_seen = now
            x1, y1, x2, y2 = p.box
            if x2 - x1 < self.min_person_w:
                continue
            if t.name and p.track_id not in busy:
                continue                              # already confirmed: save CPU
            if self.body is not None and body_budget and frame_idx - t.last_body >= self.body_every:
                t.last_body = frame_idx
                b = self.body.embed(frame, p.box)
                if b is not None:
                    body_budget -= 1
                    t.body_obs = sorted(t.body_obs + [b], key=lambda o: -o.quality)[: self.max_body]
            if frame_idx - t.last_try < self.face_every:
                continue
            t.last_try = frame_idx
            faces = self.engine.detect_embed(frame, (x1, y1, x2, y1 + int(0.7 * (y2 - y1))))
            if not faces:
                continue
            o = max(faces, key=lambda f: f.quality)
            t.obs = sorted(t.obs + [o], key=lambda f: -f.quality)[: self.max_obs]
            name, sim = self.db.match_known(o.feature[None])
            uid, usim = self.db.match_unknown(o.feature[None])
            if self.debug and now - t.last_dbg >= 1.0:
                t.last_dbg = now
                print(f"[face] person #{p.track_id}: {name} {sim:.2f} | stranger {usim:.2f} | face {o.crop.shape[1]}px")
            if uid is not None and usim >= self.unknown_thr and not (name and sim >= self.known_thr and sim > usim):
                t.unknown_id = uid                    # an already saved stranger: show the same "Unknown person #N" everywhere
            if name and sim >= self.known_thr and not (uid is not None and usim >= sim):
                # a face that looks at least as much like a saved stranger as like an enrolled person is NOT accepted
                t.hits[name] = t.hits.get(name, 0) + 1
                t.sims.append(sim)
                if t.hits[name] >= self.confirm_hits and float(np.mean(t.sims[-self.confirm_hits:])) >= self.known_thr:
                    t.name, t.sim = name, float(np.mean(t.sims[-self.confirm_hits:]))
                    print(f"{time.strftime('%H:%M:%S')} RECOGNISED: person #{p.track_id} = {name} (similarity {t.sim:.2f})")
        for pid in [k for k, t in self.tracks.items() if now - t.last_seen > 300]:
            del self.tracks[pid]                      # forget old tracks

    def label(self, track_id) -> str | None:
        """Name of an enrolled person (None for strangers and people not yet recognised)."""
        t = self.tracks.get(track_id)
        return t.name if t else None

    def display(self, track_id) -> str | None:
        """The one name used on screen and in alerts: an enrolled person's name, or "Unknown person #N" (the same N
        the alert and the dashboard use). None while the person has not been identified yet."""
        t = self.tracks.get(track_id)
        if not t:
            return None
        if t.name:
            return t.name
        if t.unknown_id is not None:
            return f"Unknown person #{t.unknown_id}"
        return None

    # ---- an object was taken ------------------------------------------------------
    def on_taken(self, frame, item: str, person_id, person_box, now: float):
        self.pending.append(_Pending(item, person_id, person_box, frame.copy(), now, now + self.wait))

    def poll(self, now: float) -> list[Decision]:
        done, keep = [], []
        for p in self.pending:
            d = self._decide(p, now, force=now >= p.deadline)
            (done if d else keep).append(d or p)
        self.pending = keep
        return done

    # ---- decision -------------------------------------------------------------------
    def _decide(self, p: _Pending, now: float, force: bool):
        t = self.tracks.get(p.person_id) if p.person_id is not None else None
        if t and t.name:
            return self._finish(p, Decision("authorised", p.item, p.person_id, name=t.name, sim=t.sim), now)
        if t and t.obs:
            feats = np.stack([o.feature for o in t.obs])
            name, sim = self.db.match_known(feats)
            _, usim = self.db.match_unknown(feats)
            if name and sim >= self.known_thr and usim < sim:
                return self._finish(p, Decision("authorised", p.item, p.person_id, name=name, sim=sim), now)
            if (t.obs[0].quality >= self.q_good and len(t.obs) >= self.min_obs) or force:
                return self._unauthorised(p, t, feats, now)
        if force or p.person_id is None:
            body = self._body_for(p, t)
            if body is not None:                      # no face, but we have a body picture: recognise by body
                return self._unauthorised_body(p, body, now)
            return self._finish(p, Decision("unidentified", p.item, p.person_id, alert=True,
                                            note="no usable face or body captured"), now, alert_key=("?", p.item))
        return None

    def _body_for(self, p: _Pending, t):
        """(features (n,768), best crop) for the person who took the object, or None."""
        if t and t.body_obs:
            return np.stack([o.feature for o in t.body_obs]), t.body_obs[0].crop
        if self.body is not None and p.person_box is not None:
            o = self.body.embed(p.frame, p.person_box)       # the frame saved at pickup time
            if o is not None:
                return o.feature[None], o.crop
        return None

    def _unauthorised(self, p: _Pending, t: _Track, feats: np.ndarray, now: float):
        crop = t.obs[0].crop
        body = self._body_for(p, t)
        face_uid, fsim = self.db.match_unknown(feats)
        body_uid, bsim = (self.db.match_unknown_body(body[0]) if body else (None, -1.0))
        note = ""
        if face_uid is not None and fsim >= self.unknown_thr:
            uid, times = face_uid, self.db.touch_unknown(face_uid, feats, crop, now)
        elif body_uid is not None and bsim >= self.body_thr and not self.db.has_face(body_uid):
            # earlier stored by body only; now we also have a face: attach it to the same record
            uid, times = body_uid, self.db.add_face_to_unknown(body_uid, feats, crop, now)
            note = f"same person as earlier body-only record (body sim {bsim:.2f}); face now saved"
        else:
            uid, times = self.db.add_unknown(feats, crop, now), 1
        if body:
            self.db.add_body(uid, body[0], body[1], now)
        _, known_sim = self.db.match_known(feats)
        d = Decision("unauthorised", p.item, p.person_id, unknown_id=uid, times_seen=times,
                     sim=max(known_sim, 0.0), alert=True, note=note, body_sim=max(bsim, 0.0))
        return self._finish(p, d, now, alert_key=(uid, p.item))

    def _unauthorised_body(self, p: _Pending, body, now: float):
        """No usable face: recognise or store the stranger by body appearance."""
        feats, crop = body
        uid, sim = self.db.match_unknown_body(feats)
        if uid is not None and sim >= self.body_thr:
            times = self.db.touch_unknown_body(uid, feats, crop, now)
            note = f"face not visible; recognised by body (sim {sim:.2f})"
        else:
            uid, times, sim = self.db.add_unknown_body_only(feats, crop, now), 1, max(sim, 0.0)
            note = "face not visible; saved by body only"
        d = Decision("unauthorised", p.item, p.person_id, unknown_id=uid, times_seen=times, sim=0.0,
                     alert=True, note=note, by_body=True, body_sim=max(sim, 0.0))
        return self._finish(p, d, now, alert_key=(uid, p.item))

    def _finish(self, p: _Pending, d: Decision, now: float, alert_key=None) -> Decision:
        if d.alert and alert_key is not None:
            last = self._last_alert.get(alert_key, -1e9)
            if p.ts - last < self.dedupe:       # compare the times the object was TAKEN
                d.alert, d.note = False, f"duplicate alert suppressed (<{int(self.dedupe)}s)"
            else:
                self._last_alert[alert_key] = p.ts
        t = self.tracks.get(p.person_id) if p.person_id is not None else None
        if t is not None and d.unknown_id is not None and not t.name:
            t.unknown_id = d.unknown_id
        d.snapshot = self._save_snapshot(p, d, p.ts)
        self.db.log_event(p.ts, d.item, d.status, d.name, d.unknown_id, d.sim, d.snapshot, d.alert)
        if d.alert and self.beep:
            threading.Thread(target=_beep, daemon=True).start()
        return d

    def _save_snapshot(self, p: _Pending, d: Decision, now: float) -> str:
        img = p.frame.copy()
        color = (0, 160, 0) if d.status == "authorised" else (0, 0, 255)
        if p.person_box:
            x1, y1, x2, y2 = p.person_box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        who = d.name or (f"unknown #{d.unknown_id}" if d.unknown_id else "unidentified")
        cv2.putText(img, f"{d.status.upper()}: {d.item} / {who}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        folder = os.path.join(self.data_dir, "snapshots")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                            + f"_{d.status}_{d.item.replace(' ', '_').replace('#', '')}.jpg")
        cv2.imwrite(path, img)
        return path


def _beep():
    try:
        import winsound
        winsound.Beep(1200, 400)
    except Exception:  # noqa: BLE001  (non-Windows or no sound device)
        print("\a", end="", flush=True)
