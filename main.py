"""IP camera -> follow (or watch) the objects you select -> who has them -> authorised or ALERT.

Run:  python enroll.py --name Priya        (once per allowed person, see README)
      python main.py                       follow mode: select objects on a frozen frame, then follow them anywhere
      python main.py --reuse-zones         reuse the objects saved in zones.json (camera must not move, objects in place)
      python main.py --mode select         watch fixed spots only (object is checked at the place you selected)
      python main.py --mode detect         auto-track phones with YOLO instead of selecting objects
      python main.py --no-faces            only report who/what, no authorised/unauthorised check
      python main.py --source file --file clip.mp4      test on a recorded video
      python main.py --device cuda:0       (on the GPU server)
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
from src.overlay import label_chip
import yaml
from dotenv import load_dotenv

from src import zones
from src.detector import Detection, Detector
from src.livefeed import LiveFeed
from src.frame_source import FrameSource
from src.pickup import PickupDetector
from src.watch import ZoneWatcher

PERSON_COLOR = (80, 200, 80)
PHONE_COLOR = (0, 140, 255)
ALERT_COLOR = (0, 0, 255)
OK_COLOR = (0, 170, 0)
BANNER_SECONDS = 5.0


def draw_persons(frame, persons, authz=None):
    for d in persons:
        x1, y1, x2, y2 = d.box
        name = authz.label(d.track_id) if authz else None
        shown = authz.display(d.track_id) if authz else None
        color = OK_COLOR if name else (ALERT_COLOR if shown else PERSON_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = shown or (authz.unidentified_text(d.track_id) if authz else "person (identifying...)")
        cv2.putText(frame, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for w in d.wrists:
            cv2.circle(frame, (int(w[0]), int(w[1])), 6, (255, 0, 255), -1)


def draw_phones(frame, phones, holders):
    for d in phones:
        x1, y1, x2, y2 = d.box
        held_by = holders.get(d.track_id)
        color = ALERT_COLOR if held_by is not None else PHONE_COLOR
        tag = f"phone #{d.track_id} {d.conf:.2f}" + (f" HELD by #{held_by}" if held_by is not None else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_zones(frame, watcher):
    for it in watcher.items:
        x1, y1, x2, y2 = it.box
        if it.state == "removed":
            color, status = ALERT_COLOR, "REMOVED"
        elif it.occupied:
            color, status = (0, 220, 255), "hand in zone"
        else:
            color, status = (255, 200, 0), "OK"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{it.name}: {status} ({it.score:.2f})", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_followed(frame, follower, holders, authz):
    for o in follower.objects:
        if len(o.trail) > 1:
            pts = [(int(x), int(y)) for x, y in o.trail]
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, (255, 200, 0), 2)
        x1, y1, x2, y2 = o.box
        poly = o.outline()
        if o.state == "carried":
            pid = holders.get(o.idx)
            who = authz.display(pid) if (authz and pid is not None) else None
            col = (0, 165, 255)
            if pid is None or o.gone > 0:                     # the holder is not in view any more
                cv2.putText(frame, f"{o.name}: left the view with a person" if o.gone > 0 else f"{o.name}: not visible",
                            (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2          # the object is hidden in the hand: no box, just a marker
            cv2.circle(frame, (cx, cy), 6, col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{o.name}: in hand of {who or ('a ' + authz.unidentified_text(pid) if authz else 'a person (identifying...)')} (hidden)",
                        (cx + 10, max(15, cy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            continue
        if o.state == "lost":
            if poly is not None:
                cv2.polylines(frame, [poly], True, (150, 150, 150), 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
            cv2.putText(frame, f"{o.name}: searching...", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2)
            continue
        pid = holders.get(o.idx)
        if pid is None:
            color, tag = (255, 200, 0), f"{o.name}: at rest"
        else:
            who = authz.display(pid) if authz else None
            color, tag = ALERT_COLOR, f"{o.name}: with {who or ('a ' + authz.unidentified_text(pid) if authz else 'a person (identifying...)')}"
        if poly is not None:
            cv2.polylines(frame, [poly], True, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def live_status(follower, pickup, authz, persons, fps, source_type, who_is):
    """What the dashboard shows next to the live picture."""
    objs = []
    for o in (follower.objects if follower is not None else []):
        pid = pickup.holders.get(o.idx) if pickup is not None else None
        if o.state == "lost":
            text = "Searching"
        elif o.state == "carried":
            text = "In hand (hidden)" if pid is not None or o.gone == 0 else "Left the view with a person"
        elif pid is not None:
            text = f"With {who_is(pid)}"
        else:
            text = "At rest"
        objs.append({"name": o.name, "state": o.state, "text": text})
    pend = [{"item": q.item, "ts": q.ts} for q in (authz.pending if authz is not None else [])]   # taken, person not identified yet
    return {"fps": round(float(fps), 1), "source": source_type, "people": len(persons), "objects": objs, "pending": pend}


def describe(d):
    if d.status == "authorised":
        return f"AUTHORISED: {d.item} taken by {d.name} (similarity {d.sim:.2f})", OK_COLOR
    if d.status == "unauthorised":
        extra = f" - seen {d.times_seen}x" if d.times_seen > 1 else ""
        how = f" ({d.note})" if d.note and d.alert else ""
        sup = f" [{d.note}]" if not d.alert else ""
        return f"ALERT: {d.item} taken by Unknown person #{d.unknown_id}{extra}{how}{sup}", ALERT_COLOR
    sup = f" [{d.note}]" if not d.alert else ""
    return f"ALERT: {d.item} taken by UNIDENTIFIED person (no usable face captured){sup}", ALERT_COLOR


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["follow", "select", "detect"], help="override config 'mode'")
    ap.add_argument("--reuse-zones", action="store_true", help="skip selection, load the saved zones file")
    ap.add_argument("--no-faces", action="store_true", help="disable the face check")
    ap.add_argument("--source", choices=["webcam", "rtsp", "file"], help="override source.type")
    ap.add_argument("--file", help="video file path (with --source file)")
    ap.add_argument("--device", help="override device, e.g. cpu or cuda:0")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.source:
        cfg["source"]["type"] = args.source
    if args.file:
        cfg["source"]["file_path"] = args.file
    if args.device:
        cfg["device"] = args.device
    mode = args.mode or cfg.get("mode", "follow")

    person_det = Detector(cfg["pose"], device=cfg["device"])            # people + wrists
    object_det = Detector(cfg["detector"], device=cfg["device"]) if mode == "detect" else None
    pcfg = dict(cfg.get("pickup", {}))
    if mode == "follow":
        pcfg["lost_frames"] = int(cfg.get("follow", {}).get("lost_frames", 90))   # object may hide in a hand
    pickup = PickupDetector(pcfg) if mode in ("follow", "detect") else None

    authz = None
    fcfg = cfg.get("faces", {})
    if fcfg.get("enabled", True) and not args.no_faces:
        from src.authz import Authorizer
        from src.facedb import FaceDB
        from src.faces import FaceEngine
        db = FaceDB(fcfg.get("db_path", "data/faces.db"), fcfg.get("data_dir", "data"))
        body = None
        if fcfg.get("body_enabled", True):
            try:
                from src.body import BodyEngine
                body = BodyEngine(fcfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[body] body re-identification unavailable ({exc}); using the face only.")
        authz = Authorizer(fcfg, db, FaceEngine(fcfg), body)
        known = db.list_known()
        if known:
            print("Enrolled:", ", ".join(f"{n} ({k})" for n, k in known))
        else:
            print("WARNING: nobody is enrolled yet, so EVERY pickup will raise an alert. "
                  "Run: python enroll.py --name <your name>")

    rt = cfg["runtime"]
    disp_w = int(rt.get("display_width", 1280))
    source = FrameSource(cfg["source"]).start()
    dcfg = cfg.get("dashboard", {})
    live = LiveFeed(dcfg.get("live_dir", "data/live"), float(dcfg.get("frame_every_sec", 0.06))) \
        if dcfg.get("live_feed", True) else None                     # frame + status for dashboard/app.py
    watcher = follower = None
    names: dict[int, str] = {}
    try:
        if mode in ("follow", "select"):
            wcfg = cfg.get("watch", {})
            ref_frame, items = zones.get_items(source, wcfg.get("zones_file", "zones.json"),
                                               args.reuse_zones, disp_w)
            if args.reuse_zones:                              # the saved outline is only where it WAS: search the whole frame
                try:
                    from src.relocate import relocate_items
                    relocate_items(ref_frame, items, fcfg.get("data_dir", "data"))
                except Exception as e:
                    print(f"Could not search for the saved objects ({e}); using their saved places.")
            try:                                              # keep pictures of the objects for the dashboard's Objects page
                from src.objects import register_objects
                register_objects(ref_frame, items, fcfg.get("db_path", "data/faces.db"), fcfg.get("data_dir", "data"))
                zones.save_items(items, wcfg.get("zones_file", "zones.json"), quiet=True)     # remembers each object's id
            except Exception as e:                            # never block tracking because of this
                print(f"Object pictures were not saved ({e}).")
            if mode == "select":
                watcher = ZoneWatcher(wcfg, items, ref_frame)
                print(f"Watching {len(items)} fixed spot(s). Press q or Esc in the window to quit.")
            else:
                from src.follow import ObjectFollower
                assist = None
                if cfg.get("follow", {}).get("assist", True):
                    try:
                        from src.assist import ObjectAssist
                        acfg = dict(cfg.get("detector", {}))
                        acfg["conf"] = cfg.get("follow", {}).get("assist_conf", 0.15)   # low: small dark things score low
                        assist = ObjectAssist(acfg, cfg["device"], ref_frame, items)
                    except Exception as e:                      # never block the run because of the helper
                        print(f"Detector assist is off ({e}).")
                follower = ObjectFollower(cfg.get("follow", {}), items, ref_frame, assist)
                names = {o.idx: o.name for o in follower.objects}
                print("Following the object(s) wherever they go. Press q or Esc in the window to quit.")
        else:
            print("Mode: detect (auto-track phones). Press q or Esc in the window to quit.")

        skip = max(1, int(rt.get("frame_skip", 1)))
        show = bool(rt.get("show_window", True)) and os.environ.get("LTS_NO_WINDOW") != "1"   # the dashboard shows the picture itself
        persons: list[Detection] = []
        phones: list[Detection] = []
        prev_holders: dict = {}
        perf = {"t": time.time(), "n": 0, "pose": 0.0, "faces": 0.0, "follow": 0.0}
        carried_taken: dict = {}
        banner = None
        n, fps, t0 = 0, 0.0, time.time()

        def who_is(pid):
            nm = authz.display(pid) if authz else None
            if pid is None:
                return "the holder (not tracked)"
            return nm or ("a " + authz.unidentified_text(pid) if authz else "a person (identifying...)")

        while True:
            ok, frame = source.read()
            if not ok:
                print("No more frames (end of file or camera timeout).")
                break
            if n % skip == 0:
                now = time.time()
                _t = time.perf_counter()
                persons = person_det.track(frame)
                perf["pose"] += time.perf_counter() - _t
                _t = time.perf_counter()
                if authz:
                    authz.observe(frame, persons, now, n)
                perf["faces"] += time.perf_counter() - _t
                _t = time.perf_counter()
                taken = []          # (item, person_id, person_box, kind)
                if watcher is not None:
                    for ev in watcher.update(frame, persons, now):
                        if ev.kind == "returned":
                            print(time.strftime("%H:%M:%S"), f"RETURNED: {ev.item} is back")
                        else:
                            taken.append((ev.item, ev.person_id, ev.person_box, ev.kind))
                else:
                    if follower is not None:
                        phones = follower.update(frame, n, persons, pickup.holders)
                        for kind, o, hid in follower.notes:
                            t = time.strftime("%H:%M:%S")
                            if kind in ("carried", "left") and hid is not None and o.idx not in pickup.holders \
                                    and now - carried_taken.get(o.idx, -1e9) > 20:
                                # the object vanished while a person was next to it, before the hand-on-object check could confirm a pickup:
                                # that is still a pickup, so it must be checked and logged (with a snapshot) like any other
                                carried_taken[o.idx] = now
                                hb = next((p.box for p in persons if p.track_id == hid), None)
                                taken.append((o.name, hid, hb, "taken out of sight"))
                            if kind == "carried":
                                print(t, f"CARRIED: {o.name} is out of sight, assumed to be with {who_is(hid)}")
                            elif kind == "reset":
                                print(t, f"RESET: {o.name} tracker had jumped away; put back on the object at rest")
                            elif kind == "back":
                                print(t, f"FOUND: {o.name} is visible again")
                            elif kind == "left":
                                print(t, f"LEFT: {o.name} left the view with {who_is(hid)}")
                        for o in follower.just_lost:
                            last = pickup.holders.get(o.idx)
                            where = f"last seen with {who_is(last)}" if last is not None else "last seen at rest"
                            print(time.strftime("%H:%M:%S"), f"LOST: {o.name} is not visible ({where})")
                    else:
                        phones = object_det.track(frame)
                    for ev in pickup.update(phones, persons, now):
                        item = names.get(ev.phone_id, f"phone #{ev.phone_id}")
                        taken.append((item, ev.person_id, ev.person_box, ev.kind))
                    cur = dict(pickup.holders)
                    for oid in set(cur) | set(prev_holders):
                        nm = names.get(oid, f"phone #{oid}")
                        if cur.get(oid) != prev_holders.get(oid):
                            if cur.get(oid) is not None:
                                print(time.strftime("%H:%M:%S"), f"HOLDER: {nm} is now with {who_is(cur[oid])}")
                            else:
                                o = next((x for x in follower.objects if x.idx == oid), None) if follower else None
                                if o is not None and o.state in ("lost", "carried"):
                                    # holder dropped but the object was never seen put down: not a confirmed put-down
                                    print(time.strftime("%H:%M:%S"),
                                          f"NOT VISIBLE: {nm} was last with {who_is(prev_holders.get(oid))} and has not been seen put down; "
                                          "will pick it up again if it reappears")
                                else:
                                    print(time.strftime("%H:%M:%S"), f"PUT DOWN: {nm} put down, now watched at its new place")
                    prev_holders = cur
                perf["follow"] += time.perf_counter() - _t
                for item, pid, pbox, kind in taken:
                    who = f"person #{pid}" if pid is not None else "nobody identified"
                    print(time.strftime("%H:%M:%S"), f"TAKEN: {item} ({who}, {kind})")
                    if authz:
                        authz.on_taken(frame, item, pid, pbox, now)
                    else:
                        banner = (time.time(), f"TAKEN: {item} by {who}", ALERT_COLOR)
                if authz:
                    for d in authz.poll(now):
                        msg, color = describe(d)
                        print(time.strftime("%H:%M:%S"), msg, f"-> {d.snapshot}")
                        banner = (time.time(), msg, color)
            n += 1
            if n % 15 == 0:
                fps = 15 / (time.time() - t0)
                t0 = time.time()
            if time.time() - perf["t"] >= 10:                 # where the time goes, in the tracker log (tuning aid)
                k = max(1, n - perf["n"])
                print(f"PERF: {fps:.1f} fps | per processed frame: people {1000 * perf['pose'] / max(1, k // skip):.0f} ms, "
                      f"faces/body {1000 * perf['faces'] / max(1, k // skip):.0f} ms, objects {1000 * perf['follow'] / max(1, k // skip):.0f} ms "
                      f"| analysing every {skip} frame(s)")
                perf.update(t=time.time(), n=n, pose=0.0, faces=0.0, follow=0.0)
            if show or live is not None:
                draw_persons(frame, persons, authz)
                if watcher is not None:
                    draw_zones(frame, watcher)
                elif follower is not None:
                    draw_followed(frame, follower, pickup.holders, authz)
                else:
                    draw_phones(frame, phones, pickup.holders)
                cv2.putText(frame, f"FPS {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if banner and time.time() - banner[0] < BANNER_SECONDS:
                    label_chip(frame, banner[1], banner[2])
                if disp_w and frame.shape[1] > disp_w:       # only the display is scaled, not the analysis
                    frame = cv2.resize(frame, None, fx=disp_w / frame.shape[1], fy=disp_w / frame.shape[1],
                                       interpolation=cv2.INTER_AREA)
                if live is not None:
                    live.publish(frame, live_status(follower, pickup, authz, persons, fps, cfg["source"]["type"], who_is))
                if show:
                    cv2.imshow("live tracking", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        if live is not None:
            live.close()
        source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
