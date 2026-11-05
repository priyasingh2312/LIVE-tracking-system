"""SQLite store for known people, unknown people (saved for future matches) and an event log."""
from __future__ import annotations

import os
import sqlite3
import time

import cv2
import numpy as np


class FaceDB:
    def __init__(self, db_path: str = "data/faces.db", data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.con = sqlite3.connect(db_path)
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS known_persons(id INTEGER PRIMARY KEY, name TEXT UNIQUE, created REAL);
        CREATE TABLE IF NOT EXISTS known_embeddings(id INTEGER PRIMARY KEY, person_id INTEGER, emb BLOB);
        CREATE TABLE IF NOT EXISTS known_body(id INTEGER PRIMARY KEY, person_id INTEGER, emb BLOB, image_path TEXT);
        CREATE TABLE IF NOT EXISTS unknown_persons(id INTEGER PRIMARY KEY, first_seen REAL, last_seen REAL, times_seen INTEGER);
        CREATE TABLE IF NOT EXISTS unknown_embeddings(id INTEGER PRIMARY KEY, unknown_id INTEGER, emb BLOB, image_path TEXT);
        CREATE TABLE IF NOT EXISTS unknown_body(id INTEGER PRIMARY KEY, unknown_id INTEGER, emb BLOB, image_path TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, item TEXT, status TEXT,
                                          name TEXT, unknown_id INTEGER, sim REAL, snapshot TEXT, alert INTEGER);
        """)
        if "name" not in [r[1] for r in self.con.execute("PRAGMA table_info(unknown_persons)")]:
            self.con.execute("ALTER TABLE unknown_persons ADD COLUMN name TEXT")      # a label you can give a stranger
            self.con.commit()
        ecols = [r[1] for r in self.con.execute("PRAGMA table_info(events)")]
        for col, ddl in (("camera", "TEXT"), ("reviewed", "INTEGER DEFAULT 0")):      # alert page: which camera, reviewed?
            if col not in ecols:
                self.con.execute(f"ALTER TABLE events ADD COLUMN {col} {ddl}")
        self.con.commit()
        self.known: dict[str, np.ndarray] = {}
        self.unknown: dict[int, np.ndarray] = {}
        self.unknown_body: dict[int, np.ndarray] = {}      # body fingerprints (768) of strangers
        self._load()

    # ---- loading ------------------------------------------------------------
    @staticmethod
    def _arr(blobs, dim=128):
        return np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs]) if blobs else np.zeros((0, dim), np.float32)

    def _load(self):
        self.known.clear()
        for pid, name in self.con.execute("SELECT id, name FROM known_persons"):
            rows = [r[0] for r in self.con.execute("SELECT emb FROM known_embeddings WHERE person_id=?", (pid,))]
            self.known[name] = self._arr(rows)
        self.unknown.clear()
        for (uid,) in self.con.execute("SELECT id FROM unknown_persons"):
            rows = [r[0] for r in self.con.execute("SELECT emb FROM unknown_embeddings WHERE unknown_id=?", (uid,))]
            self.unknown[uid] = self._arr(rows)
        self.unknown_body.clear()
        for (uid,) in self.con.execute("SELECT id FROM unknown_persons"):
            rows = [r[0] for r in self.con.execute("SELECT emb FROM unknown_body WHERE unknown_id=?", (uid,))]
            self.unknown_body[uid] = self._arr(rows, 768)

    # ---- known people ---------------------------------------------------------
    def add_known(self, name: str, feats: list[np.ndarray], crops: list[np.ndarray]) -> int:
        cur = self.con.execute("INSERT OR IGNORE INTO known_persons(name, created) VALUES(?,?)", (name, time.time()))
        pid = self.con.execute("SELECT id FROM known_persons WHERE name=?", (name,)).fetchone()[0]
        folder = os.path.join(self.data_dir, "known", name)
        os.makedirs(folder, exist_ok=True)
        start = len(os.listdir(folder))
        for i, (f, c) in enumerate(zip(feats, crops)):
            self.con.execute("INSERT INTO known_embeddings(person_id, emb) VALUES(?,?)",
                             (pid, f.astype(np.float32).tobytes()))
            cv2.imwrite(os.path.join(folder, f"{start + i + 1:02d}.jpg"), c)
        self.con.commit()
        self._load()
        return pid

    def add_known_body(self, name: str, feats: list[np.ndarray], crops: list[np.ndarray]) -> int:
        """Save body fingerprints + pictures of an enrolled person (data/known/<name>/body_NN.jpg).
        They are for the record and the dashboard only: the face alone decides who is authorised."""
        row = self.con.execute("SELECT id FROM known_persons WHERE name=?", (name,)).fetchone()
        if not row:
            raise ValueError(f"{name!r} is not enrolled")
        folder = os.path.join(self.data_dir, "known", name)
        os.makedirs(folder, exist_ok=True)
        start = len([f for f in os.listdir(folder) if f.startswith("body")])
        for i, (f, c) in enumerate(zip(feats, crops)):
            path = os.path.join(folder, f"body_{start + i + 1:02d}.jpg")
            cv2.imwrite(path, c)
            self.con.execute("INSERT INTO known_body(person_id, emb, image_path) VALUES(?,?,?)",
                             (row[0], f.astype(np.float32).tobytes(), path))
        self.con.commit()
        return len(feats)

    def delete_known(self, name: str) -> bool:
        row = self.con.execute("SELECT id FROM known_persons WHERE name=?", (name,)).fetchone()
        if not row:
            return False
        self.con.execute("DELETE FROM known_embeddings WHERE person_id=?", (row[0],))
        self.con.execute("DELETE FROM known_body WHERE person_id=?", (row[0],))
        self.con.execute("DELETE FROM known_persons WHERE id=?", (row[0],))
        self.con.commit()
        self._load()
        import shutil
        shutil.rmtree(os.path.join(self.data_dir, "known", name), ignore_errors=True)   # its saved pictures too
        return True

    def list_known(self):
        return [(n, len(g)) for n, g in self.known.items()]

    def match_known(self, feats: np.ndarray):
        """Best (name, similarity) over all enrolled people; similarity = max cosine over embeddings."""
        best_name, best = None, -1.0
        for name, gal in self.known.items():
            if len(gal) == 0:
                continue
            s = float((feats @ gal.T).max())
            if s > best:
                best_name, best = name, s
        return best_name, best

    # ---- unknown people -------------------------------------------------------
    def match_unknown(self, feats: np.ndarray):
        best_id, best = None, -1.0
        for uid, gal in self.unknown.items():
            if len(gal) == 0:
                continue
            s = float((feats @ gal.T).max())
            if s > best:
                best_id, best = uid, s
        return best_id, best

    def _save_unknown_crop(self, uid: int, crop, now: float) -> str:
        folder = os.path.join(self.data_dir, "unknown", f"unknown_{uid}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + ".jpg")
        cv2.imwrite(path, crop)
        return path

    def add_unknown(self, feats: np.ndarray, crop, now: float) -> int:
        cur = self.con.execute("INSERT INTO unknown_persons(first_seen,last_seen,times_seen) VALUES(?,?,1)", (now, now))
        uid = cur.lastrowid
        path = self._save_unknown_crop(uid, crop, now)
        for f in feats[:3]:
            self.con.execute("INSERT INTO unknown_embeddings(unknown_id, emb, image_path) VALUES(?,?,?)",
                             (uid, f.astype(np.float32).tobytes(), path))
        self.con.commit()
        self.unknown[uid] = feats[:3].astype(np.float32).copy()
        return uid

    def touch_unknown(self, uid: int, feats: np.ndarray, crop, now: float) -> int:
        self.con.execute("UPDATE unknown_persons SET last_seen=?, times_seen=times_seen+1 WHERE id=?", (now, uid))
        path = self._save_unknown_crop(uid, crop, now)
        gal = self.unknown[uid]
        for f in feats[:3]:   # keep a few diverse embeddings per unknown person (max 8)
            if len(gal) < 8 and float((gal @ f).max()) < 0.85:
                self.con.execute("INSERT INTO unknown_embeddings(unknown_id, emb, image_path) VALUES(?,?,?)",
                                 (uid, f.astype(np.float32).tobytes(), path))
                gal = np.vstack([gal, f[None]])
        self.unknown[uid] = gal
        self.con.commit()
        return self.con.execute("SELECT times_seen FROM unknown_persons WHERE id=?", (uid,)).fetchone()[0]

    # ---- unknown people: body fingerprints --------------------------------------------
    def match_unknown_body(self, feats: np.ndarray):
        best_id, best = None, -1.0
        for uid, gal in self.unknown_body.items():
            if len(gal) == 0:
                continue
            s = float((feats @ gal.T).max())
            if s > best:
                best_id, best = uid, s
        return best_id, best

    def has_face(self, uid: int) -> bool:
        return len(self.unknown.get(uid, ())) > 0

    def _save_body_crop(self, uid: int, crop, now: float) -> str:
        folder = os.path.join(self.data_dir, "unknown", f"unknown_{uid}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "body_" + time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + ".jpg")
        cv2.imwrite(path, crop)
        return path

    def add_body(self, uid: int, feats: np.ndarray, crop, now: float):
        """Remember up to 8 diverse body fingerprints for an unknown person."""
        gal = self.unknown_body.get(uid, np.zeros((0, 768), np.float32))
        path = None
        for f in feats[:3]:
            if len(gal) < 8 and (len(gal) == 0 or float((gal @ f).max()) < 0.92):
                path = path or self._save_body_crop(uid, crop, now)
                self.con.execute("INSERT INTO unknown_body(unknown_id, emb, image_path) VALUES(?,?,?)",
                                 (uid, f.astype(np.float32).tobytes(), path))
                gal = np.vstack([gal, f[None].astype(np.float32)])
        self.unknown_body[uid] = gal
        self.con.commit()

    def add_unknown_body_only(self, feats: np.ndarray, crop, now: float) -> int:
        """A stranger whose face was never seen: stored by body only (face can be added later)."""
        uid = self.con.execute("INSERT INTO unknown_persons(first_seen,last_seen,times_seen) VALUES(?,?,1)",
                               (now, now)).lastrowid
        self.unknown[uid] = np.zeros((0, 128), np.float32)
        self.unknown_body[uid] = np.zeros((0, 768), np.float32)
        self.add_body(uid, feats, crop, now)
        return uid

    def touch_unknown_body(self, uid: int, feats: np.ndarray, crop, now: float) -> int:
        self.con.execute("UPDATE unknown_persons SET last_seen=?, times_seen=times_seen+1 WHERE id=?", (now, uid))
        self.add_body(uid, feats, crop, now)
        return self.con.execute("SELECT times_seen FROM unknown_persons WHERE id=?", (uid,)).fetchone()[0]

    def add_face_to_unknown(self, uid: int, feats: np.ndarray, crop, now: float):
        """Attach the face of a person first stored by body only."""
        path = self._save_unknown_crop(uid, crop, now)
        for f in feats[:3]:
            self.con.execute("INSERT INTO unknown_embeddings(unknown_id, emb, image_path) VALUES(?,?,?)",
                             (uid, f.astype(np.float32).tobytes(), path))
        self.unknown[uid] = feats[:3].astype(np.float32).copy()
        self.con.execute("UPDATE unknown_persons SET last_seen=?, times_seen=times_seen+1 WHERE id=?", (now, uid))
        self.con.commit()
        return self.con.execute("SELECT times_seen FROM unknown_persons WHERE id=?", (uid,)).fetchone()[0]

    # ---- events ---------------------------------------------------------------
    def log_event(self, ts, item, status, name, unknown_id, sim, snapshot, alert):
        camera = os.environ.get("CAMERA_ID", "").strip()
        self.con.execute("INSERT INTO events(ts,item,status,name,unknown_id,sim,snapshot,alert,camera,reviewed) VALUES(?,?,?,?,?,?,?,?,?,0)",
                         (ts, item, status, name, unknown_id, sim, snapshot, int(alert), camera))
        self.con.commit()
