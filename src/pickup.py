"""Pickup logic: decides when a tracked object (phone) is picked up by a tracked person.

A phone is "picked up" by person P when, for `hold_frames` consecutive processed
frames, one of P's wrists is close to the phone AND the phone has moved away
from the resting position it had before (so someone just walking past, or a
phone that was already in a hand when the program started, does not fire).

State per phone track:  idle (resting) -> held (after a pickup) -> idle (released).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from src.detector import Detection


@dataclass
class PickupEvent:
    ts: float
    phone_id: int
    person_id: int
    phone_box: tuple[int, int, int, int]
    person_box: tuple[int, int, int, int]
    kind: str = "pickup"            # "pickup" | "handover" (someone else took it from the holder)


class _PhoneState:
    def __init__(self, center):
        self.rest = center          # where the phone sits when nobody touches it
        self.state = "idle"         # idle | held
        self.cand: int | None = None
        self.cand_count = 0
        self.holder: int | None = None
        self.away = 0
        self.missing = 0
        self.last_event = 0.0
        self.switch_cand: int | None = None
        self.switch_count = 0


def _dist_point_to_box(pt, box) -> float:
    px, py = pt
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


class PickupDetector:
    def __init__(self, cfg: dict):
        self.hand_margin = float(cfg.get("hand_margin", 0.8))    # x phone size
        self.hold_frames = int(cfg.get("hold_frames", 4))
        self.release_frames = int(cfg.get("release_frames", 8))
        self.move_thresh = float(cfg.get("move_thresh", 0.8))    # x phone size
        self.lost_frames = int(cfg.get("lost_frames", 15))
        self.cooldown = float(cfg.get("cooldown_sec", 3.0))
        self.require_movement = bool(cfg.get("require_movement", True))
        self.switch_frames = int(cfg.get("switch_frames", 8))     # frames before a hand-over counts
        self.states: dict[int, _PhoneState] = {}
        self.holders: dict[int, int] = {}   # phone_id -> person_id currently holding (for drawing)

    def update(self, phones: list[Detection], persons: list[Detection],
               now: float | None = None) -> list[PickupEvent]:
        now = time.time() if now is None else now
        events: list[PickupEvent] = []
        seen: set[int] = set()

        for ph in phones:
            if ph.track_id is None:
                continue
            seen.add(ph.track_id)
            st = self.states.get(ph.track_id)
            if st is None:
                st = self.states[ph.track_id] = _PhoneState(ph.center)
            st.missing = 0

            x1, y1, x2, y2 = ph.box
            size = max(x2 - x1, y2 - y1, 1)

            # nearest person wrist to this phone
            best, best_d = None, float("inf")
            for p in persons:
                if p.track_id is None:
                    continue
                for w in p.wrists:
                    d = _dist_point_to_box(w, ph.box)
                    if d < best_d:
                        best, best_d = p, d
            near = best is not None and best_d <= self.hand_margin * size

            if st.state == "idle":
                if near:
                    if best.track_id == st.cand:
                        st.cand_count += 1
                    else:
                        st.cand, st.cand_count = best.track_id, 1
                    disp = math.hypot(ph.center[0] - st.rest[0], ph.center[1] - st.rest[1]) / size
                    moved = disp >= self.move_thresh or not self.require_movement
                    if st.cand_count >= self.hold_frames and moved:
                        st.state, st.holder, st.away = "held", st.cand, 0
                        self.holders[ph.track_id] = st.cand
                        if now - st.last_event >= self.cooldown:
                            st.last_event = now
                            events.append(PickupEvent(now, ph.track_id, st.cand, ph.box, best.box))
                else:
                    st.cand, st.cand_count = None, 0
                    # slowly follow the phone while it rests untouched
                    st.rest = (0.8 * st.rest[0] + 0.2 * ph.center[0],
                               0.8 * st.rest[1] + 0.2 * ph.center[1])
            else:  # held
                if near:
                    st.away = 0
                    if best.track_id == st.holder:
                        st.switch_cand, st.switch_count = None, 0
                    else:   # someone else is closest: a hand-over once it lasts long enough
                        if best.track_id == st.switch_cand:
                            st.switch_count += 1
                        else:
                            st.switch_cand, st.switch_count = best.track_id, 1
                        if st.switch_count >= self.switch_frames:
                            st.holder = best.track_id
                            self.holders[ph.track_id] = best.track_id
                            st.switch_cand, st.switch_count = None, 0
                            events.append(PickupEvent(now, ph.track_id, best.track_id, ph.box, best.box, "handover"))
                else:
                    st.away += 1
                    if st.away >= self.release_frames:
                        st.state, st.holder = "idle", None
                        st.rest, st.cand, st.cand_count = ph.center, None, 0   # new resting place
                        st.switch_cand, st.switch_count = None, 0
                        self.holders.pop(ph.track_id, None)

        # forget phone tracks that disappeared
        for pid in list(self.states):
            if pid not in seen:
                self.states[pid].missing += 1
                if self.states[pid].missing > self.lost_frames:
                    del self.states[pid]
                    self.holders.pop(pid, None)
        return events
