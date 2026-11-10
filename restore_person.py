"""Put an enrolled person back from a saved set of face fingerprints (used when someone was deleted by mistake).

  python restore_person.py --name Priya --file data/restore_Priya.npz

Only the face fingerprints come back (no pictures), which is all the tracker needs to recognise the person. Add more samples any time
from the dashboard (People > Enrol new person with the same name)."""
from __future__ import annotations

import argparse
import time

import numpy as np
import yaml

from src.facedb import FaceDB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        fc = yaml.safe_load(f).get("faces", {})
    db = FaceDB(fc.get("db_path", "data/faces.db"), fc.get("data_dir", "data"))
    if args.name in db.known:
        raise SystemExit(f"'{args.name}' is already enrolled ({len(db.known[args.name])} samples). Nothing changed.")
    emb = np.load(args.file)["emb"].astype(np.float32)
    db.con.execute("INSERT INTO known_persons(name, created) VALUES(?,?)", (args.name, time.time()))
    pid = db.con.execute("SELECT id FROM known_persons WHERE name=?", (args.name,)).fetchone()[0]
    for e in emb:
        db.con.execute("INSERT INTO known_embeddings(person_id, emb) VALUES(?,?)", (pid, e.tobytes()))
    db.con.commit()
    print(f"Restored '{args.name}' with {len(emb)} face samples.")


if __name__ == "__main__":
    main()
