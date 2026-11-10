"""Enrol the people who ARE allowed to take the objects.

  python enroll.py --name Priya                  live window: START, turn your head slightly, STOP & SAVE
  python enroll.py --name Priya --images "D:\\photos\\priya"   enrol from a folder of photos instead
  python enroll.py --name Priya --body           add a full-body picture + fingerprint (record only; never used to authorise)
  python enroll.py --list                        show enrolled people
  python enroll.py --delete Priya                remove a person
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from src.facedb import FaceDB
from src.faces import FaceEngine
from src.frame_source import FrameSource


def _body_tools(cfg):
    """(person detector, body engine) or (None, None) if body fingerprints are switched off or unavailable."""
    fc = cfg.get("faces", {})
    if not fc.get("body_enabled", True):
        return None, None
    try:
        from src.body import BodyEngine
        from src.detector import Detector
        return Detector(cfg["pose"], device=cfg.get("device", "cpu")), BodyEngine(fc)
    except Exception as exc:  # noqa: BLE001
        print(f"[body] body fingerprints unavailable ({exc}); enrolling the face only.")
        return None, None


def _person_for_face(people, face_box):
    """The detected person whose box contains the centre of the face."""
    cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
    inside = [d for d in people if d.box[0] <= cx <= d.box[2] and d.box[1] <= cy <= d.box[3]]
    return max(inside, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), default=None)


def _button(img, rect, text, fill, active=True):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), fill if active else (110, 110, 110), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.putText(img, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def enrol_from_webcam(name, cfg, db, engine, samples, body_samples=4):
    """Capture face samples AND (when possible) body fingerprints from the camera in cfg['source'].

    Like outlining an object, collection is under your control: the window opens as a live preview, START begins collecting
    samples, STOP & SAVE ends it and saves them (CANCEL / q discards everything). SPACE toggles start/stop."""
    det, body = _body_tools(cfg)
    src = FrameSource(cfg["source"]).start()
    max_face, max_body = samples * 3, body_samples * 3                  # safety caps; you decide when to stop
    feats, crops, last = [], [], 0.0
    bfeats, bcrops, blast = [], [], 0.0
    ui = {"collecting": False, "act": None, "rects": {}, "msg": ""}
    print(f"Enrolling '{name}': a live window opened. Press START (or SPACE), slowly turn your head a little"
          + (" and stand back so your whole body is in view" if body else "") + ", then STOP & SAVE (or SPACE). CANCEL or q = discard.")

    def on_mouse(event, x, y, _f, _p):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for k, (x1, y1, x2, y2) in ui["rects"].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                ui["act"] = k

    cv2.namedWindow("enrol")
    cv2.setMouseCallback("enrol", on_mouse)
    saved = False
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                raise RuntimeError("No frames from the camera.")
            faces = engine.detect_embed(frame, min_face_px=60)
            disp = frame.copy()
            f = max(faces, key=lambda x: x.quality) if faces else None
            col = ui["collecting"]
            if f is not None:
                x1, y1, x2, y2 = f.box
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0) if col else (200, 200, 200), 2)
                cv2.putText(disp, f"quality {f.quality:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                novel = not feats or float((np.stack(feats) @ f.feature).max()) < 0.97
                if col and len(feats) < max_face and f.quality >= 0.45 and novel and time.time() - last > 0.7:
                    feats.append(f.feature)
                    crops.append(f.crop)
                    last = time.time()
                    print(f"  face {len(feats)}")
            if body is not None and (col or ui["act"] is None) and time.time() - blast > (1.5 if col else 0.4):
                people = det.track(frame)
                person = _person_for_face(people, f.box) if f is not None else (people[0] if len(people) == 1 else None)
                if person is not None:
                    cv2.rectangle(disp, person.box[:2], person.box[2:], (255, 160, 0), 2)
                    if col and len(bfeats) < max_body:
                        o = body.embed(frame, person.box)
                        if o is not None and o.quality >= 0.5:
                            bfeats.append(o.feature)
                            bcrops.append(o.crop)
                            print(f"  body {len(bfeats)}")
                blast = time.time()
            if disp.shape[1] > 1280:
                disp = cv2.resize(disp, None, fx=1280 / disp.shape[1], fy=1280 / disp.shape[1])
            H, W = disp.shape[:2]
            head = (f"{name}: COLLECTING  face {len(feats)} (aim for {samples}+)" + (f", body {len(bfeats)} (aim for {body_samples}+)" if body else "")) \
                if col else f"{name}: ready - press START when the person is in view"
            cv2.rectangle(disp, (0, 0), (W, 44), (30, 30, 30), -1)
            cv2.putText(disp, head, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255) if col else (255, 255, 255), 2)
            if ui["msg"]:
                cv2.putText(disp, ui["msg"], (10, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2)
            bw, bh, y = 230, 46, H - 58
            ui["rects"] = {"toggle": (10, y, 10 + bw, y + bh), "cancel": (W - 10 - 160, y, W - 10, y + bh)}
            _button(disp, ui["rects"]["toggle"], "STOP & SAVE (SPACE)" if col else "START (SPACE)", (40, 40, 200) if col else (60, 160, 40))
            _button(disp, ui["rects"]["cancel"], "CANCEL (q)", (90, 90, 90))
            cv2.imshow("enrol", disp)
            k = cv2.waitKey(1) & 0xFF
            act, ui["act"] = ui["act"], None
            if k == 32:
                act = "toggle"
            if k in (ord("q"), 27):
                act = "cancel"
            if act == "cancel":
                print("Cancelled, nothing saved.")
                return
            if act == "toggle":
                if not col:
                    ui["collecting"], ui["msg"] = True, ""
                    print("Collecting...")
                elif not feats:
                    ui["msg"] = "No face captured yet: face the camera, then press STOP & SAVE again."
                else:
                    break
        saved = True
    finally:
        src.stop()
        cv2.destroyAllWindows()
    if not saved:
        return
    db.add_known(name, feats, crops)
    msg = f"Saved {len(feats)} face samples for '{name}'"
    if bfeats:
        db.add_known_body(name, bfeats, bcrops)
        msg += f" and {len(bfeats)} body samples"
    elif body is not None:
        msg += " (no body picture was taken now: that is fine, they are added automatically the next time the tracker recognises this person)"
    print(msg + ".")


def add_body_photo(name, cfg, db):
    """Add a full-body picture and fingerprint to an enrolled person. Display/record only: the face decides authorisation."""
    if name not in db.known:
        raise SystemExit(f"'{name}' is not enrolled yet. Run: python enroll.py --name {name}")
    det, body = _body_tools(cfg)
    if body is None:
        raise SystemExit("Body fingerprints are switched off or unavailable (faces.body_enabled).")
    src = FrameSource(cfg["source"]).start()
    print("Stand so your whole body is in view. SPACE = save, q = quit.")
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                raise RuntimeError("No frames from the camera.")
            people = det.track(frame)
            best = max(people, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), default=None)
            disp = frame.copy()
            if best:
                cv2.rectangle(disp, best.box[:2], best.box[2:], (0, 255, 0), 2)
            cv2.putText(disp, f"{name}: SPACE = save body picture, q = quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if disp.shape[1] > 1280:
                disp = cv2.resize(disp, None, fx=1280 / disp.shape[1], fy=1280 / disp.shape[1])
            cv2.imshow("enrol body", disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                print("Nothing saved.")
                return
            if k == 32:
                o = body.embed(frame, best.box) if best else None
                if o is None:
                    print("  no usable body seen (too small or cut off), try again")
                    continue
                db.add_known_body(name, [o.feature], [o.crop])
                print("Saved a body picture.")
                return
    finally:
        src.stop()
        cv2.destroyAllWindows()


def enrol_from_images(name, folder, db, engine):
    feats, crops = [], []
    paths = [p for ext in ("jpg", "jpeg", "png", "bmp") for p in glob.glob(os.path.join(folder, f"*.{ext}"))]
    for p in sorted(paths):
        img = cv2.imread(p)
        faces = engine.detect_embed(img, min_face_px=40) if img is not None else []
        if not faces:
            print(f"  no usable face in {os.path.basename(p)} - skipped")
            continue
        f = max(faces, key=lambda x: (x.box[2] - x.box[0]))
        feats.append(f.feature)
        crops.append(f.crop)
    if not feats:
        raise SystemExit("No faces found. Use clear, front-facing photos.")
    db.add_known(name, feats, crops)
    print(f"Saved {len(feats)} face(s) from {len(paths)} image(s) for '{name}'.")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name")
    ap.add_argument("--images", help="folder of photos to enrol from (instead of the webcam)")
    ap.add_argument("--source", choices=["webcam", "rtsp"], help="override source.type (rtsp uses RTSP_URL)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--body", action="store_true", help="add a full-body picture for an enrolled person (dashboard only)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.source:
        cfg["source"]["type"] = args.source
    fc = cfg.get("faces", {})
    db = FaceDB(fc.get("db_path", "data/faces.db"), fc.get("data_dir", "data"))

    if args.list:
        people = db.list_known()
        print("\n".join(f"{n}: {k} sample(s)" for n, k in people) or "Nobody enrolled yet.")
        return
    if args.delete:
        print("Deleted." if db.delete_known(args.delete) else "No such person.")
        return
    if not args.name:
        ap.error("--name is required (or use --list / --delete)")
    if args.body:
        add_body_photo(args.name, cfg, db)
        return
    engine = FaceEngine(fc)
    if args.images:
        enrol_from_images(args.name, args.images, db, engine)
    else:
        enrol_from_webcam(args.name, cfg, db, engine, args.samples)


if __name__ == "__main__":
    main()
