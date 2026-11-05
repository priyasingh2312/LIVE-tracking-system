"""Live Tracking System dashboard (Streamlit).

Run from the project folder:   streamlit run dashboard/app.py

It only READS what the tracker writes (data/faces.db, data/snapshots, data/live), so it can be opened or closed at
any time without affecting tracking. Start the tracker separately:   python main.py --source rtsp
Set DASHBOARD_PASSWORD in .env to ask for a password (the pages show faces of enrolled and unknown people).
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:                                   # python-dotenv is optional here
    pass

try:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as _f:
        CFG = yaml.safe_load(_f) or {}
except OSError:
    CFG = {}
FCFG = CFG.get("faces", {})
DCFG = CFG.get("dashboard", {})
TITLE = DCFG.get("title", "Live Tracking System")
SUBTITLE = DCFG.get("subtitle", "Object tracking, authorisation checks and re-identification of unauthorised pickups")
DB_PATH = ROOT / FCFG.get("db_path", "data/faces.db")
DATA_DIR = ROOT / FCFG.get("data_dir", "data")
LIVE_DIR = ROOT / DCFG.get("live_dir", "data/live")
ALIVE_SEC = 6                                       # tracker counts as running if its status is this fresh

GOOD, WARN, BAD, BLUE, TEAL = "#167A3B", "#9A6200", "#B42318", "#1F5FBF", "#0F9D8F"
RESULT = {"authorised": ("Authorised", "ok"), "unauthorised": ("Unauthorised", "bad"), "unidentified": ("Unidentified", "warn")}

try:                                                # newer Streamlit retires use_container_width
    from packaging.version import Version
    STRETCH = {"width": "stretch"} if Version(st.__version__) >= Version("1.50") else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
[data-testid="stSidebarCollapseButton"] {display: none;}
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1600px;}
:root {--teal: #0F9D8F; --teal-d: #0B8478; --teal-bg: #E3F5F2; --navy: #0B3A6E; --navy-2: #1D4F86; --navy-3: #12467F;}
.stApp {background: #F3F7FB;}

/* cards */
[data-testid="stVerticalBlockBorderWrapper"] {background: #ffffff; border: 1px solid #E3E8F0 !important; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(15,27,45,.05), 0 4px 14px rgba(15,27,45,.04);}
.kpi {background: #fff; border: 1px solid #E3E8F0; border-left: 5px solid var(--c, #1F5FBF); border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,27,45,.05), 0 4px 14px rgba(15,27,45,.04);}
.camcard {border: 1px solid #E3E8F0; border-radius: 10px; padding: 8px 12px; margin-top: 6px; background: #fff;}
.camcard.on {border-color: #0F9D8F; background: #EEF9F7;}
.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 14px; margin-bottom: 6px;}
@media (max-width: 900px) {.kpi-grid {grid-template-columns: 1fr;}}
.kpi .l {color: #5B6778; font-size: .74rem; font-weight: 700; letter-spacing: .8px; text-transform: uppercase;}
.kpi .v {color: #0F1B2D; font-size: 2rem; font-weight: 700; line-height: 1.15; margin-top: 2px;}
.kpi .s {color: #7A8696; font-size: .8rem; margin-top: 2px;}

/* top bar and chips */
.topbar {display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 4px 0 14px 0; margin-bottom: 10px; border-bottom: 1px solid #E3E8F0;}
.topbar h1 {margin: 0; padding: 0; font-size: 1.6rem; font-weight: 700; color: #0F1B2D; letter-spacing: .2px;}
.topbar .sub {color: #5B6778; font-size: .9rem; margin-top: 2px;}
.chips {display: flex; gap: 8px; flex-wrap: wrap; align-items: center;}
.chip {display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px;
    font-size: .76rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap;}
.chip::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor;}
.chip.ok::before {animation: pulse 1.8s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(22,122,59,.45);} 70% {box-shadow: 0 0 0 6px rgba(22,122,59,0);} 100% {box-shadow: 0 0 0 0 rgba(22,122,59,0);}}
.chip.ok {background: #E6F6EC; color: #167A3B; border-color: #BFE5CC;}
.chip.warn {background: #FFF4DD; color: #9A6200; border-color: #F3DDA8;}
.chip.bad {background: #FDE8E8; color: #B42318; border-color: #F5C2C0;}
.chip.idle {background: #EEF1F6; color: #5B6778; border-color: #DCE2EC;}

/* live view */
.panel-title {font-weight: 700; font-size: .95rem; color: #0F1B2D; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;}
.counts {display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 2px 0;}
.count {background: #F1F5FB; border: 1px solid #DDE6F3; border-radius: 10px; padding: 4px 12px; font-size: .85rem; color: #1B2430;}
.count b {color: #0F1B2D; margin-left: 4px;}
.empty {text-align: center; padding: 56px 12px; color: #5B6778;}
.empty h3 {color: #0F1B2D; margin-bottom: 4px;}
.quiet {color: #7A8696; font-size: .85rem; margin-top: 10px;}

/* activity feed */
.feed-item {display: flex; gap: 10px; padding: 9px 0; border-top: 1px solid #EDF0F5; align-items: center;}
.feed-item:first-of-type {border-top: none;}
.feed-item img {width: 84px; height: 56px; object-fit: cover; border-radius: 8px; border: 1px solid #E3E8F0; flex: none;}
.feed-item .th {position: relative; display: block; width: 84px; height: 56px; flex: none; cursor: zoom-in;}
.feed-item .th:hover {z-index: 999999;}
.feed-item .th:hover img {position: absolute; right: -4px; top: -4px; width: 480px; max-width: 60vw; height: auto;
    border: 1px solid #3B82F6; border-radius: 6px; background: #fff; box-shadow: 0 8px 28px rgba(0,0,0,.35); object-fit: contain; pointer-events: none;}
.dtitle {font-weight: 700; font-size: 1.05rem; color: #0F1B2D; margin-top: 10px;}
.dsub {color: #667085; margin: 2px 0 10px 0; font-size: .92rem;}
.drow {display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 10px 2px; border-bottom: 1px solid #EEF1F6; font-size: .9rem;}
.drow span {color: #667085;} .drow b {font-weight: 600; color: #0F1B2D; text-align: right;}
.rchip {display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; border: 1px solid #E3E8F0; background: #F2F4F7; color: #3D4A5C;}
.rchip.ok {background: #E6F6EC; color: #167A3B; border-color: #BFE5CC;} .rchip.bad {background: #FDECEA; color: #B42318; border-color: #F6C9C4;}
.rchip.warn {background: #FFF4D6; color: #8A5A00; border-color: #F2DDA1;}
.stage {display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; min-height: 380px;
    border-radius: 14px; color: #fff; padding: 24px; position: relative; background: linear-gradient(180deg, #123B6D 0%, #0B2A52 100%);}
.stage h3 {color: #fff; margin: 14px 0 4px; font-size: 18px;}
.stage p {color: #B8C9E0; margin: 0; font-size: 14px;}
.stage .src {position: absolute; left: 14px; bottom: 12px; font-size: 12px; color: #B8C9E0;}
.selcam {color: #2563A8; font-size: 13px; margin-top: 6px;}
.card-sub {color: #667085; font-size: 12px; margin-top: -6px;}
.logrow {position: relative; display: grid; grid-template-columns: 170px 150px 1fr 70px; gap: 12px; align-items: center; padding: 9px 14px;
    border-bottom: 1px solid #E6EAF0; font-size: 14px; background: #fff;}
.logrow.head {font-size: 12px; font-weight: 600; color: #667085; text-transform: uppercase; background: #F6F8FB; border-radius: 8px 8px 0 0;}
.logrow:not(.head):hover {background: #F0F7FF; z-index: 999999; cursor: zoom-in;}
.logrow .pop {display: none; position: absolute; right: 12px; top: 100%; width: 480px; max-width: 60vw; height: auto; margin-top: 4px;
    border: 1px solid #3B82F6; border-radius: 6px; background: #fff; box-shadow: 0 8px 28px rgba(0,0,0,.35); pointer-events: none;}
.logrow:hover .pop {display: block;}
.logrow .nosnap {color: #98A2B3; font-size: 12px;}
.logwrap {border: 1px solid #E6EAF0; border-radius: 8px; margin-bottom: 8px;}
.feed-item .ph {width: 84px; height: 56px; border-radius: 8px; background: #EEF1F6; flex: none;}
.feed-item .t {font-weight: 650; font-size: .92rem; color: #0F1B2D; line-height: 1.2;}
.feed-item .e {font-size: .8rem; font-weight: 600;}
.feed-item .e.bad {color: #B42318;} .feed-item .e.ok {color: #167A3B;} .feed-item .e.warn {color: #9A6200;}
.feed-item .w {font-size: .74rem; color: #7A8696;}

/* people */
.pcard {background: #fff; border: 1px solid #E3E8F0; border-radius: 12px; padding: 12px; margin-bottom: 10px;}
.pcard .imgs {display: flex; gap: 8px; margin-bottom: 8px;}
.pcard img {height: 110px; border-radius: 8px; border: 1px solid #E3E8F0; object-fit: cover; background: #EEF1F6;}
.dpic {width: 100%; height: 130px; object-fit: contain; border-radius: 8px; border: 1px solid #E3E8F0; background: #EEF1F6;}
.pgrid {display: flex; flex-wrap: wrap; gap: 8px;}
.pgrid img {height: 150px; border-radius: 8px; border: 1px solid #E3E8F0; object-fit: cover; background: #EEF1F6;}
[class*="st-key-ptile_"] {position: relative;}
[class*="st-key-open_"] {position: absolute; inset: 0 0 10px 0; width: 100%; height: auto; margin: 0; z-index: 2;}
[class*="st-key-open_"], [class*="st-key-open_"] > div, [class*="st-key-open_"] .stButton {height: 100%;}
[class*="st-key-open_"] button {width: 100%; height: 100%; opacity: 0; cursor: pointer; min-height: 0;}
[class*="st-key-ptile_"]:hover .pcard {border-color: #0F9D8F; background: #F4FBFA;}
.pcard .n {font-weight: 700; color: #0F1B2D;} .pcard .m {color: #5B6778; font-size: .8rem;}

/* tabs, buttons */
[data-baseweb="tab-list"] {gap: 6px;} [data-baseweb="tab"] {font-weight: 600; padding: 8px 14px;}
[data-baseweb="tab-highlight"] {background-color: var(--teal) !important;}
button[role="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] p {color: var(--teal-d) !important;}
[data-testid="stMain"] .stButton button[kind="primary"] {background: var(--teal); border-color: var(--teal); color: #fff;}
[data-testid="stMain"] .stButton button[kind="primary"]:hover {background: var(--teal-d); border-color: var(--teal-d);}
[data-testid="stMain"] [data-testid="stDownloadButton"] button {background: #fff; color: var(--teal-d); border: 1px solid var(--teal);}
[data-testid="stMain"] [data-testid="stDownloadButton"] button:hover {background: var(--teal-bg);}

/* sidebar */
[data-testid="stSidebar"] {background: var(--navy) !important; border-right: none;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span:not(.chip), [data-testid="stSidebar"] li {color: #DCE5F3 !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #9DB6D8 !important;}
[data-testid="stSidebar"] .stButton button {background: #1A4577; border: 1px solid #2D62A3; color: #EAF2FC; border-radius: 9px; font-weight: 600; justify-content: flex-start;}
[data-testid="stSidebar"] .stButton button:hover {background: var(--navy-3) !important; border-color: #4F7BC4; color: #fff;}
[data-testid="stSidebar"] .stButton button[kind="primary"] {background: var(--teal) !important; border-color: var(--teal) !important; color: #fff;}
[data-testid="stSidebar"] .stButton button p {color: #EAF2FC !important;}
.brand {display: flex; align-items: center; gap: 10px; margin: 2px 0 14px 0;}
.brand .name {font-weight: 700; font-size: 1.1rem; color: #FFFFFF; line-height: 1.15;}
.brand .tag {color: #9DB6D8; font-size: .76rem;}
.side-label {font-size: .7rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #9DB6D8; margin: 20px 0 6px 0;}
[data-testid="stForm"] {border: none; padding: 0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- small helpers
def esc(x) -> str:
    return html.escape(str(x))


def chip(text: str, kind: str = "idle") -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def kpi(label_: str, value, sub: str = "", color: str = BLUE) -> str:
    return (f'<div class="kpi" style="--c:{color}"><div class="l">{esc(label_)}</div>'
            f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>')


def ago(ts: float) -> str:
    d = max(0, time.time() - ts)
    if d < 10:
        return "just now"
    if d < 90:
        return f"{int(d)} s ago"
    if d < 5400:
        return f"{int(d // 60)} min ago"
    if d < 86400:
        return f"{int(d // 3600)} h ago"
    return datetime.fromtimestamp(ts).strftime("%d %b %H:%M")


def frag(every):
    """st.fragment(run_every=...) with fallbacks for older Streamlit versions."""
    f = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if f is None:
        return lambda fn: fn
    try:
        return f(run_every=every)
    except TypeError:
        return f


def resolve(p) -> Path | None:
    """A path stored in the database (maybe with Windows slashes, relative to the project) as a real file."""
    if not p:
        return None
    q = Path(str(p).replace("\\", "/"))
    for cand in (q, ROOT / q, DATA_DIR / q):
        try:
            if cand.is_file():
                return cand
        except OSError:
            pass
    return None


@st.cache_data(show_spinner=False, max_entries=400)
def _thumb(path_str: str, mtime: float, width: int) -> str:
    from PIL import Image
    im = Image.open(path_str).convert("RGB")
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumb(path: Path | None, width: int = 480) -> str | None:
    if path is None:
        return None
    try:
        return _thumb(str(path), path.stat().st_mtime, width)
    except Exception:                                 # unreadable image: show a grey box instead
        return None


# ---------------------------------------------------------------- database (read only)
def _connect(write: bool = False):
    if not DB_PATH.exists():
        return None
    try:
        if write:
            return sqlite3.connect(str(DB_PATH), timeout=15)
        return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None


def _migrate():
    """Older databases: add the 'name' column for unknown people (a label you can give a stranger)."""
    con = _connect(write=True)
    if con is None:
        return
    try:
        con.execute("CREATE TABLE IF NOT EXISTS known_body(id INTEGER PRIMARY KEY, person_id INTEGER, emb BLOB, image_path TEXT)")
        con.commit()
        cols = [r[1] for r in con.execute("PRAGMA table_info(unknown_persons)")]
        if cols and "name" not in cols:
            con.execute("ALTER TABLE unknown_persons ADD COLUMN name TEXT")
            con.commit()
        ecols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
        for col, ddl in (("camera", "TEXT"), ("reviewed", "INTEGER DEFAULT 0")):
            if ecols and col not in ecols:
                con.execute(f"ALTER TABLE events ADD COLUMN {col} {ddl}")
                con.commit()
    except sqlite3.Error:
        pass
    finally:
        con.close()


_migrate()


def query(sql: str, params=()) -> pd.DataFrame:
    con = _connect()
    if con is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, con, params=params)
    except Exception:                                 # table not created yet (tracker never ran)
        return pd.DataFrame()
    finally:
        con.close()


def load_events(days: int | None = None) -> pd.DataFrame:
    sql = ("SELECT e.id, e.ts, e.item, e.status, e.name, e.unknown_id, e.sim, e.snapshot, e.alert, e.camera, e.reviewed, "
           "u.times_seen AS times_seen, u.name AS uname FROM events e LEFT JOIN unknown_persons u ON u.id = e.unknown_id ")
    params: tuple = ()
    if days is not None:
        sql += "WHERE e.ts >= ? "
        params = (time.time() - days * 86400,)
    df = query(sql + "ORDER BY e.ts DESC", params)
    if not df.empty:
        on = query("SELECT label, name FROM objects")                    # the tracker logs 'object 5'; show the name you gave it
        if not on.empty:
            df["item"] = df["item"].map(dict(zip(on["label"], on["name"]))).fillna(df["item"])
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(datetime.now().astimezone().tzinfo)
        # how many pickups this unknown person has made up to and including this one (1 = first time), not how often the face was seen
        allev = query("SELECT id, unknown_id FROM events WHERE unknown_id IS NOT NULL ORDER BY ts ASC, id ASC")
        if not allev.empty:
            allev["nth"] = allev.groupby("unknown_id").cumcount() + 1
            df["times_seen"] = df["id"].map(dict(zip(allev["id"], allev["nth"])))
        else:
            df["times_seen"] = float("nan")
    return df


def what_happened(r) -> tuple[str, str, str]:
    """(headline, result label, kind) for an event row."""
    label_, kind = RESULT.get(r["status"], (str(r["status"]).title(), "idle"))
    item = r["item"] or "object"
    if r["status"] == "authorised":
        sim = f" (similarity {r['sim']:.2f})" if pd.notna(r["sim"]) and r["sim"] > 0 else ""
        return f"{r['name']} took {item}{sim}", label_, kind
    if r["status"] == "unauthorised":
        times = int(r["times_seen"]) if pd.notna(r.get("times_seen")) else 0
        more = f" (pickup no. {times})" if times > 0 else ""
        who = f"Unknown person #{int(r['unknown_id'])}" if pd.notna(r["unknown_id"]) else "Unknown person"
        if pd.notna(r.get("uname")) and str(r.get("uname")).strip():
            who = f"{r['uname']} (unauthorised, unknown #{int(r['unknown_id'])})"
        return f"{who} took {item}{more}", label_, kind
    return f"{item} taken by someone who could not be identified", label_, kind


def live_state() -> tuple[dict, bool, Path | None]:
    status, running = {}, False
    try:
        with open(LIVE_DIR / "status.json", "r", encoding="utf-8") as f:
            status = json.load(f)
        running = bool(status.get("running")) and time.time() - float(status.get("ts", 0)) < ALIVE_SEC
    except (OSError, ValueError):
        pass
    frame = LIVE_DIR / "frame.jpg"
    return status, running, (frame if frame.is_file() else None)


st.markdown("<style>\n"+'\n.cam {border: 1px solid #E3E8F0; border-radius: 12px; padding: 11px 13px; margin-bottom: 10px; background: #fff;}\n.cam.sel {border-color: #0F9D8F; background: #EAF8F5;}\n.cam .n {font-weight: 700; color: #0F1B2D; display: flex; align-items: center; white-space: nowrap; padding-right: 68px;}\n.cam .st {font-size: .8rem; color: #7A8696; display: flex; align-items: center; gap: 6px; margin: 3px 0 4px 0;}\n.dot {width: 8px; height: 8px; border-radius: 50%; background: #98A2B3; display: inline-block; flex: none;}\n.dot.ok {background: #167A3B;} .dot.bad {background: #D92D20;} .dot.wait {background: #E8A317;}\n.card-title {font-weight: 700; font-size: 1.05rem; color: #0F1B2D; white-space: nowrap;}\n.card-title small {color: #1F5FBF; font-weight: 600; margin-left: 8px; font-size: .86rem;}\n[class*="st-key-camsel_"], [class*="st-key-camact_"] {position: relative;}\n[class*="st-key-use_"] {position: absolute; inset: 0 0 10px 0; width: 100%; height: auto; margin: 0; z-index: 2;}\n[class*="st-key-use_"] button {width: 100%; height: 100%; opacity: 0; cursor: pointer; min-height: 0;}\n[class*="st-key-camed_"], [class*="st-key-camdel_"] {position: absolute; top: 8px; width: auto; z-index: 3;}\n[class*="st-key-camed_"] {right: 42px;} [class*="st-key-camdel_"] {right: 10px;}\n[class*="st-key-camed_"] button, [class*="st-key-camdel_"] button {background: transparent !important; border: none !important; box-shadow: none !important;\n    min-height: 0; height: 30px; width: 30px; padding: 0; display: flex; align-items: center; justify-content: center;}\n[class*="st-key-camed_"] button p, [class*="st-key-camdel_"] button p {font-size: 0; line-height: 0; margin: 0;}\n[class*="st-key-camed_"] button p::before, [class*="st-key-camdel_"] button p::before {width: 18px; height: 18px;}\n[class*="st-key-camed_"] button {color: #1F5FBF !important;} [class*="st-key-camdel_"] button {color: #D92D20 !important;}\n[class*="st-key-camed_"] button:hover, [class*="st-key-camdel_"] button:hover {background: rgba(15,157,143,.12) !important;}\n[class*="st-key-camsel_"]:hover .cam.pick {border-color: #0F9D8F; background: #F4FBFA;}\n.cam .n .cnm {font-weight: 500; color: #3D4A5C; margin-left: 8px; overflow: hidden; text-overflow: ellipsis;}\n.cam .n .radio.off {background: #fff; border-color: #B9C4D4; box-shadow: none;}\n.cam .n .radio {display: inline-block; width: 18px; height: 18px; border-radius: 50%; border: 2px solid #0F9D8F; background: #0F9D8F;\n    box-shadow: inset 0 0 0 3px #fff; margin-right: 10px; flex: none;}\n.infonote {background: #EAF1FD; border: 1px solid #CFE0FA; color: #1F4F9F; border-radius: 10px; padding: 10px 12px; font-size: .9rem; font-weight: 500;\n    display: flex; gap: 8px; align-items: center; margin: 6px 0 12px;}\n.infonote svg {flex: none;}\n.st-key-btn_addcam button {white-space: nowrap; padding-left: 8px; padding-right: 8px;}\n.st-key-btn_addcam button p {overflow: visible; text-overflow: clip; font-size: .86rem;}\n[class*="st-key-camed_"] button p::before {content: ""; display: inline-block; width: 22px; height: 22px; flex: none; background-color: currentColor; -webkit-mask: url("data:image/svg+xml;utf8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22black%22 stroke-width=%221.7%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cpath d=%22M4 20h4L19 9l-4-4L4 16z%22/%3E%3Cpath d=%22M13.500 6.500l4 4%22/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml;utf8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22black%22 stroke-width=%221.7%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cpath d=%22M4 20h4L19 9l-4-4L4 16z%22/%3E%3Cpath d=%22M13.500 6.500l4 4%22/%3E%3C/svg%3E") center / contain no-repeat;}\n[class*="st-key-camdel_"] button p::before {content: ""; display: inline-block; width: 22px; height: 22px; flex: none; background-color: currentColor; -webkit-mask: url("data:image/svg+xml;utf8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22black%22 stroke-width=%221.7%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cpath d=%22M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3%22/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml;utf8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22black%22 stroke-width=%221.7%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cpath d=%22M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3%22/%3E%3C/svg%3E") center / contain no-repeat;}\n'+"</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- password gate (optional)
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
if PASSWORD and not st.session_state.get("auth"):
    _l, _m, _r = st.columns([1, 1.2, 1])
    with _m:
        st.markdown(f'<div class="topbar"><div><h1>{esc(TITLE)}</h1><div class="sub">Sign in to continue</div></div></div>',
                    unsafe_allow_html=True)
        with st.form("login"):
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                if pw == PASSWORD:
                    st.session_state["auth"] = True
                    st.rerun()
                st.error("Wrong password.")
    st.stop()


# ---------------------------------------------------------------- sidebar
PAGES = {"dash": "Dashboard View", "log": "Alert log", "objects": "Objects", "people": "People"}
if "page" not in st.session_state:
    st.session_state["page"] = "dash"


def _go(page: str):
    st.session_state["page"] = page


def app_header():
    _s, run, _f = live_state()
    now = datetime.now().astimezone().strftime("%d %b %Y \u00b7 %H:%M")
    st.markdown(f'<div class="apphead"><div class="l"><span class="nm">{esc(TITLE)}</span><span class="sb">{esc(SUBTITLE)}</span></div>'
                f'<div class="r">{chip("Tracker running", "ok") if run else chip("Tracker stopped", "idle")}<span class="dt">{now}</span></div></div>',
                unsafe_allow_html=True)


UI_FILE = DATA_DIR / "ui.json"


def _load_pin() -> bool:
    try:
        return bool(json.loads(UI_FILE.read_text(encoding="utf-8")).get("pinned"))
    except (OSError, ValueError):
        return False


def _toggle_pin():
    st.session_state["pinned"] = not st.session_state["pinned"]
    try:
        UI_FILE.parent.mkdir(parents=True, exist_ok=True)
        UI_FILE.write_text(json.dumps({"pinned": st.session_state["pinned"]}), encoding="utf-8")
    except OSError:
        pass


if "pinned" not in st.session_state:
    st.session_state["pinned"] = _load_pin()

_ICONS = {
    "dash": "M3 3h8v8H3zM13 3h8v5h-8zM13 10h8v11h-8zM3 13h8v8H3z",
    "log": "M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10 21a2 2 0 0 0 4 0",
    "objects": "M21 8l-9-5-9 5v8l9 5 9-5zM3 8l9 5 9-5M12 13v8",
    "people": "M16 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M12 10a4 4 0 1 0 0-8 4 4 0 0 0 0 8M22 20v-2a4 4 0 0 0-3-3.9M16 2.1a4 4 0 0 1 0 7.8",
    "open": "M13 17l5-5-5-5M6 17l5-5-5-5",
    "close": "M11 17l-5-5 5-5M18 17l-5-5 5-5",
}


def _mask(d: str) -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.8" '
           f'stroke-linecap="round" stroke-linejoin="round"><path d="{d}"/></svg>')
    u = 'url("data:image/svg+xml;utf8,' + quote(svg, safe="") + '") center / contain no-repeat'
    return f"-webkit-mask: {u}; mask: {u};"


def sidebar_css(pinned: bool) -> str:
    SB = 'section[data-testid="stSidebar"]'
    c = ['[data-testid="stHeader"] {display: none !important;}',
         '[data-testid="stAppViewContainer"] {padding-top: 58px;}',
         '.block-container {padding-top: 1.4rem !important;}',
         '.apphead {position: fixed; top: 0; left: 0; right: 0; height: 58px; z-index: 1000001; background: #fff; border-bottom: 1px solid #E3E8F0; '
         'display: flex; align-items: center; justify-content: space-between; padding: 0 24px;}',
         '.apphead .l {display: flex; align-items: center; gap: 16px;} .apphead .nm {font-weight: 800; font-size: 1.25rem; color: #0F1B2D;}',
         '.apphead .sb {color: #5B6778; font-size: .95rem; padding-left: 16px; border-left: 1px solid #E3E8F0; line-height: 26px;}',
         '.apphead .r {display: flex; align-items: center; gap: 18px;} .apphead .dt {color: #5B6778; font-size: .85rem;}',
         f'{SB} {{min-width: 0 !important; transform: none !important; transition: width .22s ease; overflow: hidden;}}',
         '[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], '
         '[data-testid="stSidebarHeader"] {display: none !important;}',
         f'{SB} * {{box-sizing: border-box;}}',
         '[data-testid="stSidebarContent"] {overflow-x: hidden !important; padding: 0 !important;}',
         '[data-testid="stSidebarUserContent"] {margin: 0 !important; width: 100% !important;}',
         '[data-testid="stSidebarUserContent"] {padding: 18px 16px !important;}',
         '.brand {margin: 2px 0 18px 0; white-space: nowrap;} .brand .bt {overflow: hidden; min-width: 0;} .brand .tag {overflow: hidden; text-overflow: ellipsis;}',
         '.brand .mark {flex: none; width: 36px; height: 36px; border-radius: 10px; background: #0F9D8F; color: #fff; display: flex; '
         'align-items: center; justify-content: center; font-weight: 800; font-size: .95rem;}',
         '.rdot {display: none; width: 10px; height: 10px; border-radius: 50%; margin: 8px auto; background: #98A2B3;}',
         '.rdot.ok {background: #34D399; box-shadow: 0 0 0 3px rgba(52,211,153,.25);}',
         f'{SB} .stButton button p {{display: flex; align-items: center; gap: 12px; white-space: nowrap;}}',
         f'{SB} .stButton button p::before {{content: ""; flex: none; width: 20px; height: 20px; background-color: currentColor;}}',
         f'{SB} .stButton button {{min-height: 42px;}}']
    for k, d in _ICONS.items():
        if k in ("open", "close"):
            if (k == "close") == pinned:
                c.append(f'{SB} .st-key-nav_pin button p::before {{{_mask(d)}}}')
            continue
        c.append(f'{SB} .st-key-nav_{k} button p::before {{{_mask(d)}}}')
    c.append(f'{SB} .st-key-nav_pin button {{background: transparent !important; border-color: transparent !important; margin-bottom: 6px;}}')
    if pinned:
        c.append(f'{SB} {{width: 280px !important; max-width: 280px !important;}}')
    else:
        c.append(f'{SB} {{width: 68px !important; max-width: 68px !important;}}')
        c.append(f'{SB}:hover {{width: 280px !important; max-width: 280px !important; box-shadow: 6px 0 24px rgba(0,0,0,.18);}}')
        q = f'{SB}:not(:hover)'
        c += [f'{q} [data-testid="stSidebarUserContent"] {{padding: 18px 10px !important;}}',
              f'{q} .brand .bt, {q} .side-label, {q} .chip, {q} [data-testid="stCaptionContainer"] {{display: none !important;}}',
              f'{q} .brand {{justify-content: center;}}',
              f'{q} .rdot {{display: block;}}',
              f'{q} .stButton, {q} .stButton > button {{width: 100% !important;}}',
              f'{q} .stButton button {{padding: 0 !important; justify-content: center;}}',
              f'{q} .stButton button p {{font-size: 0; gap: 0; justify-content: center;}}']
    return "<style>" + "\n".join(c) + "</style>"


app_header()
with st.sidebar:
    st.markdown(sidebar_css(st.session_state["pinned"]), unsafe_allow_html=True)
    st.button("Collapse menu" if st.session_state["pinned"] else "Expand menu", key="nav_pin", on_click=_toggle_pin, **STRETCH)
    for key, text in PAGES.items():
        st.button(text, key=f"nav_{key}", on_click=_go, args=(key,), **STRETCH,
                  type="primary" if st.session_state["page"] == key else "secondary")
    st.markdown('<div class="side-label">Tracker</div>', unsafe_allow_html=True)
    _st, _run, _ = live_state()
    st.markdown(chip("Running" if _run else "Not running", "ok" if _run else "idle")
                + f'<div class="rdot {"ok" if _run else "idle"}" title="{"Running" if _run else "Not running"}"></div>', unsafe_allow_html=True)
    st.caption("Start it from the Dashboard View page (Camera and control)" if not _run
               else f"Source: {_st.get('source', '?')}, {_st.get('fps', 0)} frames per second")
    if PASSWORD:
        st.markdown('<div class="side-label">Account</div>', unsafe_allow_html=True)
        if st.button("Sign out", **STRETCH):
            st.session_state["auth"] = False
            st.rerun()


def topbar(chips: str = ""):
    pass


# ---------------------------------------------------------------- page: dashboard
CAM_ICON = ('<svg viewBox="0 0 24 24" width="46" height="46" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="M16 10l6-3v10l-6-3z"/></svg>')


def _selected_cam():
    cams, sel = load_cams()
    c = next((x for x in cams if x["id"] == sel), None)
    return c, sel


@frag(1)
def live_card():
    status, running, frame = live_state()
    cam, sel = _selected_cam()
    nm = f'<small>{esc(sel)}</small>' if sel else ""
    with st.container(border=True):
        st.markdown(f'<div class="card-title">Camera view {nm}</div>', unsafe_allow_html=True)
        if frame is not None and running:
            st.image(frame.read_bytes(), **STRETCH)
            if running:
                objs = status.get("objects", [])
                counts = "".join(f'<div class="count">{esc(o["name"])}<b>{esc(o["text"])}</b></div>' for o in objs)
                counts += f'<div class="count">People in view<b>{int(status.get("people", 0))}</b></div>'
                st.markdown(f'<div class="counts">{counts}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="quiet">Showing the last picture from the tracker, which has stopped.</div>', unsafe_allow_html=True)
        else:
            src = esc(cam_where(cam)) if cam else ""
            st.markdown(f'<div class="stage">{CAM_ICON}<h3>Tracker is not running</h3><p>Select a camera on the right, then press Start. Objects are outlined fresh each time you press Start.</p>'
                        f'<span class="src">{src}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="selcam">Selected camera: {esc(sel) if sel else "none"}</div>', unsafe_allow_html=True)


def feed_html(df: pd.DataFrame, limit: int = 8) -> str:
    if df.empty:
        return '<div class="quiet">No pickups recorded yet.</div>'
    out = []
    for _, r in df.head(limit).iterrows():
        headline, label_, kind = what_happened(r)
        uri = thumb(resolve(r["snapshot"]), 480)
        th = f'<span class="th"><img src="{uri}"></span>' if uri else '<div class="ph"></div>'
        repeat = " (repeat, no new alert)" if r["status"] != "authorised" and not r["alert"] else ""
        out.append(f'<div class="feed-item">{th}<div><div class="t">{esc(headline)}</div>'
                   f'<div class="e {kind}">{esc(label_)}{esc(repeat)}</div><div class="w">{esc(ago(r["ts"]))}</div></div></div>')
    return "".join(out)


def _today_events():
    today = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    ev = load_events(days=30)
    day = ev[ev["dt"] >= today] if not ev.empty else ev
    return ev, day


@frag(3)
def kpi_row():
    _ev, day = _today_events()
    n = len(day)
    auth = int((day["status"] == "authorised").sum()) if n else 0
    alerts = int((day["alert"] == 1).sum()) if n else 0
    strangers = query("SELECT COUNT(*) AS c FROM unknown_persons")
    enrolled = query("SELECT COUNT(*) AS c FROM known_persons")
    status, running, _ = live_state()
    watched = len(status.get("objects", [])) if running else 0
    cards = [
        kpi("Objects watched", watched if running else "-", "tracker running" if running else "tracker stopped", BLUE),
        kpi("Pickups today", n, "all results", TEAL),
        kpi("Authorised", auth, "enrolled people", GOOD),
        kpi("Alerts", alerts, "not authorised", BAD),
        kpi("Unknown people", int(strangers["c"].iloc[0]) if not strangers.empty else 0,
            f"{int(enrolled['c'].iloc[0]) if not enrolled.empty else 0} enrolled", WARN),
    ]
    st.markdown('<div class="kpi-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def _go_log():
    st.session_state["page"] = "log"


@frag(3)
def feed_card():
    ev, _day = _today_events()
    cam, sel = _selected_cam()
    with st.container(border=True):
        h1, h2 = st.columns([1.3, 1])
        h1.markdown(f'<div class="card-title">Recent pickups</div><div class="card-sub">{esc(sel)}{" - " + esc(cam.get("name","")) if cam and cam.get("name") else ""}</div>',
                    unsafe_allow_html=True)
        h2.button("View all \u2192", key="btn_viewall", on_click=_go_log, type="tertiary")
        st.markdown(feed_html(ev, 6), unsafe_allow_html=True)


@frag(3)
def chart_card():
    _ev, day = _today_events()
    with st.container(border=True):
        st.markdown('<div class="panel-title"><span>Pickups per hour today</span></div>', unsafe_allow_html=True)
        if len(day) == 0:
            st.markdown('<div class="quiet">Nothing yet today.</div>', unsafe_allow_html=True)
        else:
            d = day.copy()
            d["hour"] = d["dt"].dt.hour
            piv = d.pivot_table(index="hour", columns="status", values="id", aggfunc="count", fill_value=0)
            piv = piv.reindex(range(0, 24), fill_value=0)
            order = [s for s in ("authorised", "unauthorised", "unidentified") if s in piv.columns]
            colors = {"authorised": GOOD, "unauthorised": BAD, "unidentified": "#D9A400"}
            st.bar_chart(piv[order], color=[colors[s] for s in order], height=220)


def kcontainer(key, **kw):
    return st.container(key=key, **kw)


# ---------------------------------------------------------------- camera list + start / stop
CAM_FILE = DATA_DIR / "cameras.json"          # stays on this PC (the address may contain a password); data/ is git-ignored
STATE_FILE = DATA_DIR / "run_state.json"
LOG_FILE = DATA_DIR / "tracker.log"
NOFLAG = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ZONES_FILE = ROOT / (CFG.get("watch", {}) or {}).get("zones_file", "zones.json")


def load_cams() -> tuple[list, str]:
    try:
        d = json.loads(CAM_FILE.read_text(encoding="utf-8"))
        return [c for c in d.get("cameras", []) if c.get("id")], d.get("selected", "")
    except (OSError, ValueError, AttributeError):
        return [], ""


def save_cams(cams: list, selected: str):
    try:
        CAM_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CAM_FILE.with_name("cameras.tmp")
        tmp.write_text(json.dumps({"selected": selected, "cameras": cams}, indent=2), encoding="utf-8")
        os.replace(tmp, CAM_FILE)
    except OSError:
        pass


def cam_where(c: dict) -> str:                    # never show user or password
    return f"Webcam {c.get('index', 0)}" if c.get("kind") == "webcam" else f"{c.get('host', '?')}:{c.get('port', 554)} - RTSP"


def next_cam_id(cams: list) -> str:
    nums = [int(m.group(1)) for c in cams for m in [re.fullmatch(r"CAM (\d+)", c["id"])] if m]
    return f"CAM {(max(nums) if nums else 0) + 1:02d}"


def build_rtsp(host, port, user, pw, path) -> str:
    if "%40" in user or "%40" in pw:                 # typed already encoded (wts%4012345): decode once, then encode below
        user, pw = unquote(user), unquote(pw)
    auth = ""
    if user:
        auth = quote(user, safe="") + ((":" + quote(pw, safe="")) if pw else "") + "@"
    return f"rtsp://{auth}{host.strip()}:{int(port)}/{path.strip().lstrip('/')}"


def rtsp_parts(url: str):
    p = urlparse(url) if url else None
    if not p:
        return "", 554, "", "", "stream1"
    return (p.hostname or "", p.port or 554, unquote(p.username or ""), unquote(p.password or ""),
            (p.path or "/stream1").lstrip("/") + (("?" + p.query) if p.query else ""))


def reachable(c: dict) -> str:
    """'' if the camera's address answers on its port, else the reason (a wrong network is the usual cause)."""
    import socket
    if c.get("kind") != "rtsp":
        return ""
    try:
        socket.create_connection((c["host"], int(c["port"])), timeout=3).close()
        return ""
    except OSError:
        return (f"Cannot reach {c['host']}:{c['port']} from this PC. The camera is off, the address is wrong, "
                "or this PC is on a different network (check you are on the camera's Wi-Fi or LAN).")


def test_camera(c: dict) -> tuple[bool, str]:
    """Read one picture in a separate process, so a dead address cannot hang the dashboard."""
    why = reachable(c)
    if why:
        return False, why
    if c["kind"] == "webcam":
        code = ("import cv2,sys\ni=int(sys.argv[1])\nc=cv2.VideoCapture(i,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(i)\n"
                "ok,f=c.read()\nprint('OK %dx%d' % (f.shape[1], f.shape[0]) if ok else 'FAIL')")
        arg, env = str(int(c["index"])), None
    else:
        code = ("import cv2,sys\nc=cv2.VideoCapture(sys.argv[1],cv2.CAP_FFMPEG)\nok=False\n"
                "for _ in range(40):\n    ok,f=c.read()\n    if ok: break\n"
                "print('OK %dx%d' % (f.shape[1], f.shape[0]) if ok else 'FAIL')")
        arg, env = c["url"], dict(os.environ, OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp")   # same as the tracker
    web = c["kind"] == "webcam"
    try:
        r = subprocess.run([sys.executable, "-c", code, arg], capture_output=True, text=True, timeout=30, env=env,
                           creationflags=NOFLAG)
    except subprocess.TimeoutExpired:
        return False, ("The webcam did not answer (timed out). Close other apps that use it." if web else
                       "No answer from the camera (timed out). Check the IP address, port and that the PC can reach it.")
    if r.stdout.strip().startswith("OK"):
        return True, ("Webcam works. Got a picture (" if web else "Connected. Got a picture (") + r.stdout.strip()[3:] + ")."
    return False, ("Could not read a picture. Try another webcam number (0, 1, 2) and close other apps that use the camera." if web
                   else "Could not read a picture. Check the IP, port, user, password and stream name.")


def pid_alive(pid) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                                 text=True, timeout=8, creationflags=NOFLAG).stdout.lower()
        except Exception:                           # noqa: BLE001
            return False
        return f'"{pid}"' in out and "python" in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15, creationflags=NOFLAG)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:                               # noqa: BLE001
        pass


@st.cache_resource
def ctl() -> dict:
    """One state dict for every browser tab of this dashboard server. A tracker left running by an earlier
    dashboard is picked up again; closing the dashboard (Ctrl+C) also stops a tracker it started."""
    c = {"pid": None, "proc": None, "cam": "", "started": 0.0, "log": None}
    try:
        d = json.loads(STATE_FILE.read_text())
        if pid_alive(d.get("pid")):
            c.update(pid=d["pid"], cam=d.get("cam", ""), started=float(d.get("started", 0)))
    except (OSError, ValueError):
        pass
    import atexit

    def _bye():
        try:
            if c["proc"] is not None and c["proc"].poll() is None:
                kill_pid(c["proc"].pid)
                STATE_FILE.unlink(missing_ok=True)
        except Exception:                           # noqa: BLE001
            pass
    atexit.register(_bye)
    return c


def owned_running() -> bool:
    c = ctl()
    if c["proc"] is not None:
        if c["proc"].poll() is None:
            return True
        c["proc"], c["pid"], c["died"] = None, None, True       # it stopped by itself
    elif c["pid"] and pid_alive(c["pid"]):
        return True
    return False


def start_tracker(cam: dict, reselect: bool = True):
    if (ctl().get("enrol") is not None) and ctl()["enrol"].poll() is None:
        st.session_state["cam_msg"] = ("error", "An enrolment is running. Wait until it finishes, then start.")
        return
    why = reachable(cam)
    if why:
        st.session_state["cam_msg"] = ("error", "Could not start. " + why)
        return
    st.session_state["cam_msg"] = (None, "")
    stop_tracker()
    c = ctl()
    env = dict(os.environ, CAMERA_ID=cam["id"], CAMERA_NAME=cam.get("name", ""), LTS_NO_WINDOW="1")        # saved with the objects
    cmd = [sys.executable, "-u", "main.py", "--source", "webcam" if cam["kind"] == "webcam" else "rtsp"]
    if not reselect:
        cmd.append("--reuse-zones")                                  # keep the objects from the last run, skip outlining
    if cam["kind"] == "webcam":
        env["WEBCAM_INDEX"] = str(int(cam["index"]))
    else:
        env["RTSP_URL"] = cam["url"]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    c["log"] = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
    c["proc"] = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=c["log"], stderr=subprocess.STDOUT, creationflags=NOFLAG)
    c.update(pid=c["proc"].pid, cam=cam["id"], started=time.time())
    try:
        STATE_FILE.write_text(json.dumps({"pid": c["pid"], "cam": cam["id"], "started": c["started"]}))
    except OSError:
        pass


def stop_tracker():
    c = ctl()
    p = c["proc"]
    if p is not None and p.poll() is None:
        kill_pid(p.pid)
        try:
            p.wait(8)
        except subprocess.TimeoutExpired:
            p.kill()
    elif c["pid"] and pid_alive(c["pid"]):
        kill_pid(c["pid"])
    c.update(proc=None, pid=None, cam="", died=False)
    if c["log"]:
        try:
            c["log"].close()
        except OSError:
            pass
        c["log"] = None
    STATE_FILE.unlink(missing_ok=True)
    try:                                             # the picture of the last run must not look live
        (LIVE_DIR / "status.json").write_text(json.dumps({"ts": time.time(), "running": False, "objects": []}))
    except OSError:
        pass


def _use_cam(cid):
    cams, _sel = load_cams()
    if owned_running():
        st.session_state["cam_msg"] = ("error", "Stop the tracker first (Stop button below), then select another camera.")
    elif any(c["id"] == cid for c in cams):
        save_cams(cams, cid)
        st.session_state["cam_msg"] = ("success", f"{cid} selected. Press Start to use it.")


def _toggle_cam_edit():
    st.session_state["cam_edit"] = not st.session_state.get("cam_edit")
    st.session_state["cam_target"] = None


def _edit_cam(cid):
    if owned_running():
        st.session_state["cam_msg"] = ("error", "Stop the tracker first (Stop button below), then edit the camera.")
        return
    st.session_state["cam_edit"], st.session_state["cam_target"] = True, cid
    st.session_state["cam_msg"] = (None, "")


def _ask_del(cid):
    if owned_running():
        st.session_state["cam_msg"] = ("error", "Stop the tracker first (Stop button below), then remove the camera.")
        return
    st.session_state["cam_del"] = cid


def cam_status() -> tuple[str, str]:
    _s, run, _f = live_state()
    if owned_running():
        return ("Receiving frames", "ok") if run else ("Waiting for frames", "wait")
    return ("Receiving frames", "ok") if run else ("Stopped", "off")


def control_card():
    with st.container(border=True):
        ch1, ch2 = st.columns([0.95, 1.15])
        cams0, _s0 = load_cams()
        ch1.markdown(f'<div class="card-title">Cameras<small>({len(cams0)})</small></div>', unsafe_allow_html=True)
        ch2.button("Close" if st.session_state.get("cam_edit") else "+ Add camera", key="btn_addcam", type="primary",
                   on_click=_toggle_cam_edit, **STRETCH)

        @frag(3)
        def cam_item():
            cams, sel = load_cams()
            t, d = cam_status()
            if len(cams) > 1:
                q = st.text_input("Find camera", placeholder="Find camera ...", key="cam_find",
                                  label_visibility="collapsed").strip().lower()
                if q:
                    cams = [c for c in cams if q in (c["id"] + " " + str(c.get("name", "")) + " " + cam_where(c)).lower()]
            if not cams:
                st.markdown('<div class="cam"><div class="n"><span class="radio off"></span>No camera yet</div>'
                            '<div class="st">Press + Add camera to add the first one.</div></div>', unsafe_allow_html=True)
            for c in cams:
                ck = c["id"].replace(" ", "_")
                ttl = esc(c["id"]) + (f' <span class="cnm">{esc(c["name"])}</span>' if c.get("name") and c["name"].strip().lower() != c["id"].lower() else "")
                active = c["id"] == sel
                with kcontainer(("camact_" if active else "camsel_") + ck):
                    if active:
                        st.markdown(f'<div class="cam sel"><div class="n"><span class="radio"></span>{ttl}</div>'
                                    f'<div class="st"><span class="dot {d}"></span>{esc(t)}</div>'
                                    f'<div class="st">{esc(cam_where(c))}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="cam pick"><div class="n"><span class="radio off"></span>{ttl}</div>'
                                    f'<div class="st">{esc(cam_where(c))}</div></div>', unsafe_allow_html=True)
                        if st.button("Select " + c["id"], key=f"use_{ck}"):
                            _use_cam(c["id"])
                            st.rerun()
                    if st.button("Edit", key=f"camed_{ck}"):
                        _edit_cam(c["id"])
                        st.rerun()
                    st.button("Remove", key=f"camdel_{ck}", on_click=_ask_del, args=(c["id"],))
            dcid = st.session_state.get("cam_del")
            if dcid:
                st.warning(f"Remove {dcid} from the camera list?")
                y1, y2 = st.columns(2)
                if y1.button("Remove", key="cam_del_yes", type="primary", **STRETCH):
                    left = [x for x in load_cams()[0] if x["id"] != dcid]
                    save_cams(left, sel if sel != dcid else (left[0]["id"] if left else ""))
                    st.session_state["cam_del"] = None
                    st.session_state["cam_msg"] = ("success", f"{dcid} removed.")
                    st.rerun()
                if y2.button("Cancel", key="cam_del_no", **STRETCH):
                    st.session_state["cam_del"] = None
                    st.rerun()
        cam_item()

        if st.session_state.get("cam_edit"):
            if owned_running():
                st.caption("Stop the tracker first (Stop button below), then change the camera.")
            else:
                cams, sel = load_cams()
                tid = st.session_state.get("cam_target")
                tgt = next((x for x in cams if x["id"] == tid), None) if tid else None
                h0, p0, u0, w0, path0 = rtsp_parts(tgt.get("url", "") if tgt and tgt["kind"] == "rtsp" else "")
                n0 = (tgt or {}).get("name", "")
                st.markdown(f"**Edit {tid}**" if tgt else "**Add a camera**")
                if tgt:
                    is_web = tgt["kind"] == "webcam"
                else:
                    is_web = st.radio("Camera type", ["IP camera (RTSP)", "Webcam (this PC)"], horizontal=True,
                                      key="cam_kind", label_visibility="collapsed").startswith("Webcam")
                fk = (tid or "new") + ("w" if is_web else "r")
                with st.form("camera_form_" + fk):
                    f_name = st.text_input("Camera name (optional)", value=n0, placeholder="e.g. Main entrance", key="fn_" + fk)
                    if is_web:
                        f_idx = st.number_input("Webcam number", 0, 9, int(tgt["index"]) if tgt else 0, step=1, key="fi_" + fk,
                                                help="0 = the built-in or first USB webcam, 1 = the second one")
                    else:
                        f_host = st.text_input("Camera IP address", value=h0, placeholder="192.168.1.50", key="fh_" + fk)
                        c1, c2 = st.columns(2)
                        f_port = c1.number_input("Port", 1, 65535, int(p0), step=1, key="fp_" + fk)
                        f_path = c2.text_input("Stream", value=path0, key="fs_" + fk, help="Tapo: stream1 (HD) or stream2 (light)")
                        f_user = st.text_input("User name", value=u0, key="fu_" + fk)
                        f_pw = st.text_input("Password", type="password", key="fw_" + fk,
                                             placeholder="leave empty to keep the saved one" if w0 else "")
                    b1, b2 = st.columns(2)
                    do_test = b1.form_submit_button("Test", **STRETCH)
                    do_save = b2.form_submit_button("Save", type="primary", **STRETCH)
                if do_test or do_save:
                    cam = None
                    if is_web:
                        cam = {"kind": "webcam", "index": int(f_idx)}
                    elif not f_host.strip() or not f_path.strip():
                        st.session_state["cam_msg"] = ("error", "Enter the camera IP address and the stream name.")
                    else:
                        cam = {"kind": "rtsp", "host": f_host.strip(), "port": int(f_port),
                               "url": build_rtsp(f_host, f_port, f_user, f_pw or (w0 if f_user == u0 else ""), f_path)}
                    if cam is not None:
                        if do_test:
                            with st.spinner("Connecting ..."):
                                ok, msg = test_camera(cam)
                            st.session_state["cam_msg"] = ("success" if ok else "error", msg)
                        else:
                            cam["name"] = f_name.strip()
                            if tgt:
                                cam["id"] = tid
                                cams = [cam if c["id"] == tid else c for c in cams]
                                msg = f"{tid} updated and selected. Press Start to use it."
                            else:
                                cam["id"] = next_cam_id(cams)
                                cams.append(cam)
                                msg = f"{cam['id']} added to the camera list and selected. Press Start to use this camera."
                            save_cams(cams, cam["id"])
                            st.session_state["cam_msg"] = ("success", msg)
                            st.session_state["cam_edit"], st.session_state["cam_target"] = False, None
                            st.rerun()
        c_ = ctl()
        if c_.pop("died", False) and not owned_running():
            try:
                tail = [x for x in LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()][-1]
            except (OSError, IndexError):
                tail = ""
            st.session_state["cam_msg"] = ("error", "Could not start. " + tail[:200])
        kind, msg = st.session_state.get("cam_msg", (None, ""))
        if kind:
            getattr(st, kind)(msg)

    info = ('<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.5"/></svg>')
    st.markdown(f'<div class="infonote">{info}<span>Select a camera to update the feed and alerts.</span></div>', unsafe_allow_html=True)
    cams, sel = load_cams()
    chosen = next((c for c in cams if c["id"] == sel), None)
    on = owned_running()
    have_zones = (ROOT / "zones.json").is_file()
    same = st.checkbox("Use the same objects as last time", key="reuse_objs", disabled=not have_zones,
                       help="Skips outlining and tracks the objects from your last run again. Use it only if the camera has not "
                            "moved and the objects are still where they were. Untick it to outline new objects.")
    b1, b2 = st.columns(2)
    if on:
        if b1.button("Restart", key="btn_restart", disabled=chosen is None, **STRETCH):
            start_tracker(chosen, not same)
            st.rerun()
    elif b1.button("Start", key="btn_start", type="primary", disabled=chosen is None, **STRETCH):
        start_tracker(chosen, not same)
        st.rerun()
    if b2.button("Stop", key="btn_stop", disabled=not on, **STRETCH):
        stop_tracker()
        st.rerun()
    if chosen is None:
        st.caption("Add the camera address first (Edit camera).")


def page_dash():
    status, running, _ = live_state()
    topbar(chip("Tracker running", "ok") if running else chip("Tracker stopped", "idle"))
    kpi_row()
    st.write("")
    c1, c2, c3 = st.columns([5, 3.2, 3], gap="medium")
    with c1:
        live_card()
    with c2:
        feed_card()
    with c3:
        control_card()
    st.write("")
    chart_card()


# ---------------------------------------------------------------- page: alert log
def _who(r) -> str:
    if r["status"] == "authorised":
        return str(r["name"] or "-")
    if pd.notna(r["unknown_id"]):
        nm = str(r.get("uname") or "").strip()
        return f"{nm} (unknown #{int(r['unknown_id'])})" if nm else f"Unknown person #{int(r['unknown_id'])}"
    return "Not identified"


def _review_status(r) -> str:
    if r["status"] == "authorised":
        return "-"
    return "Reviewed" if int(r["reviewed"] or 0) else "Unreviewed"


def _set_reviewed(event_id: int, value: int):
    con = _connect(write=True)
    if con is None:
        st.session_state["log_msg"] = ("error", "Could not open the database for writing.")
        return
    try:
        con.execute("UPDATE events SET reviewed=? WHERE id=?", (int(value), int(event_id)))
        con.commit()
    except sqlite3.Error as e:
        st.session_state["log_msg"] = ("error", f"Could not save: {e}")
    finally:
        con.close()


def _detail_row(label: str, value: str):
    st.markdown(f'<div class="drow"><span>{esc(label)}</span><b>{value}</b></div>', unsafe_allow_html=True)


def alert_details(r):
    with st.container(border=True):
        st.markdown('<div class="card-title">Alert details</div>', unsafe_allow_html=True)
        headline, label_, kind = what_happened(r)
        st.markdown(f'<div class="dtitle">{esc(r["item"] or "object")}</div><div class="dsub">{esc(headline)}</div>', unsafe_allow_html=True)
        path = resolve(r["snapshot"])
        if path is not None:
            st.image(str(path), **STRETCH)
        else:
            st.markdown('<div class="quiet">The snapshot file is not available.</div>', unsafe_allow_html=True)
        times = int(r["times_seen"]) if pd.notna(r.get("times_seen")) else 1
        _detail_row("Timestamp", esc(r["dt"].strftime("%d %b %Y \u00b7 %H:%M:%S")))
        _detail_row("Camera", esc(str(r["camera"]) if pd.notna(r["camera"]) and str(r["camera"]).strip() else "-"))
        _detail_row("Person", esc(_who(r)))
        _detail_row("Result", f'<span class="rchip {kind}">{esc(label_)}</span>')
        _detail_row("Count", str(max(times, 1)))
        if r["status"] != "authorised":
            done = bool(int(r["reviewed"] or 0))
            _detail_row("Status", f'<span class="rchip {"ok" if done else "idle"}">{"Reviewed" if done else "Unreviewed"}</span>')
            if done:
                st.button("Mark as unreviewed", key="rev_undo", on_click=_set_reviewed, args=(int(r["id"]), 0), **STRETCH)
            else:
                st.button("Mark reviewed", key="rev_do", type="primary", on_click=_set_reviewed, args=(int(r["id"]), 1), **STRETCH)
        kind_, msg_ = st.session_state.pop("log_msg", (None, ""))
        if kind_ and msg_:
            getattr(st, kind_)(msg_)


def page_log():
    topbar()
    ev = load_events()
    if ev.empty:
        st.markdown('<div class="empty"><h3>No pickups recorded yet</h3>They appear here once the tracker has run.</div>', unsafe_allow_html=True)
        return
    with st.container(border=True):
        a, b, c = st.columns([2, 2, 2])
        results = a.multiselect("Result", list(RESULT), default=list(RESULT), format_func=lambda s: RESULT[s][0])
        first = ev["dt"].min().date()
        rng = b.date_input("Dates", value=(max(first, date.today() - timedelta(days=7)), date.today()))
        who = c.text_input("Person or object contains")
    d = ev[ev["status"].isin(results)]
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        d = d[(d["dt"].dt.date >= rng[0]) & (d["dt"].dt.date <= rng[1])]
    if who:
        s = who.lower()
        d = d[d["name"].fillna("").str.lower().str.contains(s) | d["item"].fillna("").str.lower().str.contains(s)
              | d["unknown_id"].astype("Int64").astype(str).str.contains(s)]
    d = d.reset_index(drop=True)
    if d.empty:
        st.markdown('<div class="quiet">Nothing matches these filters.</div>', unsafe_allow_html=True)
        return
    LIMIT = 200
    shown = d.head(LIMIT)
    table = pd.DataFrame({
        "Preview": [thumb(resolve(p), 120) for p in shown["snapshot"]],
        "Timestamp": shown["dt"].dt.strftime("%d %b %Y, %H:%M:%S"),
        "Camera": [str(c) if pd.notna(c) and str(c).strip() else "-" for c in shown["camera"]],
        "Object": shown["item"].fillna("object"),
        "Person": [_who(r) for _, r in shown.iterrows()],
        "Result": [what_happened(r)[1] for _, r in shown.iterrows()],
        "Status": [_review_status(r) for _, r in shown.iterrows()],
    })
    left, right = st.columns([3, 1.35], gap="medium")
    with left:
        with st.container(border=True):
            rng_txt = f"{d['dt'].min():%d %b %Y} - {d['dt'].max():%d %b %Y}"
            st.markdown(f'<div class="card-title">{len(d)} pickups<small>{rng_txt}</small></div>', unsafe_allow_html=True)
            kw = {}
            try:
                import inspect
                if "row_height" in inspect.signature(st.dataframe).parameters:
                    kw["row_height"] = 46
            except (TypeError, ValueError):
                pass
            sel = st.dataframe(table, hide_index=True, height=min(480, 46 * len(table) + 50), key="log_table",
                               on_select="rerun", selection_mode="single-row", column_config={
                                   "Preview": st.column_config.ImageColumn("Preview", width="small"),
                                   "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                                   "Person": st.column_config.TextColumn("Person", width="medium"),
                                   "Status": st.column_config.TextColumn("Status", width="small")},
                               **STRETCH, **kw)
            st.caption(f"Showing the latest {len(shown)} of {len(d)}. Tick a row to see it on the right."
                       + (" The CSV has all of them." if len(d) > LIMIT else ""))
            st.download_button("Download CSV", table.drop(columns=["Preview"]).to_csv(index=False).encode("utf-8"),
                               "pickups.csv", "text/csv")
    rows = list(sel.selection.rows) if sel is not None and getattr(sel, "selection", None) else []
    if rows and rows[0] < len(shown):
        pick = int(shown.iloc[rows[0]]["id"])
        st.session_state["log_sel"] = pick
    else:
        pick = st.session_state.get("log_sel")
    cur = shown[shown["id"] == pick] if pick is not None else shown.iloc[0:0]
    r = (cur if not cur.empty else shown).iloc[0]
    with right:
        alert_details(r)


# ---------------------------------------------------------------- enrol a new person (opens the camera window on this PC)
ENROL_LOG = DATA_DIR / "enrol.log"


def enrol_running() -> bool:
    p = ctl().get("enrol")
    return p is not None and p.poll() is None


def start_enrol(name: str, cam: dict):
    why = reachable(cam)
    if why:
        st.session_state["enrol_msg"] = ("error", "Could not start. " + why)
        return
    env = dict(os.environ)
    cmd = [sys.executable, "-u", "enroll.py", "--name", name, "--source", "webcam" if cam["kind"] == "webcam" else "rtsp"]
    if cam["kind"] == "webcam":
        env["WEBCAM_INDEX"] = str(int(cam["index"]))
    else:
        env["RTSP_URL"] = cam["url"]
    ENROL_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(ENROL_LOG, "w", encoding="utf-8", errors="replace")
    ctl()["enrol"] = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=NOFLAG)
    ctl()["enrol_name"] = name
    st.session_state["enrol_msg"] = (None, "")


def enrol_panel():
    """Button + form to enrol a new person; the camera window opens on this PC and captures face and body samples."""
    h1, h2 = st.columns([3, 1.2])
    h2.button("Close" if st.session_state.get("enrol_open") else "+ Enrol new person", key="btn_enrol", type="primary",
              on_click=lambda: st.session_state.update(enrol_open=not st.session_state.get("enrol_open")), **STRETCH)

    @frag(2)
    def status():
        c = ctl()
        p = c.get("enrol")
        if p is not None and p.poll() is None:
            st.info(f"Enrolling {c.get('enrol_name', '')}: a window opened on this PC. Look at the camera, turn your head slowly "
                    "and step back so your whole body is in view. It closes by itself when done (q cancels).")
        elif p is not None:
            try:
                lines = [x for x in ENROL_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
            except OSError:
                lines = []
            last = lines[-1] if lines else ""
            ok = p.returncode == 0 and last.startswith("Saved")
            st.session_state["enrol_msg"] = ("success" if ok else "error", last if ok else ("Enrolment did not finish. " + last)[:300])
            c["enrol"] = None
            st.rerun()
    status()

    if st.session_state.get("enrol_open") and not enrol_running():
        cams, sel = load_cams()
        cam = next((x for x in cams if x["id"] == sel), None)
        _s, run, _f = live_state()
        with st.container(border=True):
            st.markdown("**Enrol a new person**")
            if cam is None:
                st.caption("Add and select a camera first (Dashboard View, Cameras).")
            elif owned_running() or run:
                st.warning("Stop the tracker first (the camera is in use), then enrol.")
            else:
                st.caption(f"Uses the selected camera: {cam['id']} ({cam_where(cam)}). Face and body samples are saved; "
                           "the face alone decides who is authorised.")
                nm = st.text_input("Name", key="enrol_name_in", placeholder="e.g. Priya Singh", max_chars=40).strip()
                bad = nm and not re.fullmatch(r"[A-Za-z0-9 _.\-]{1,40}", nm)
                if bad:
                    st.error("Use letters, numbers, spaces, dot, dash or underscore only.")
                if st.button("Start enrolment", key="btn_enrol_go", type="primary", disabled=not nm or bool(bad)):
                    start_enrol(nm, cam)
                    st.session_state["enrol_open"] = False
                    st.rerun()
    kind, msg = st.session_state.get("enrol_msg", (None, ""))
    if kind and msg:
        getattr(st, kind)(msg)


# ---------------------------------------------------------------- people tiles: face on the tile, all pictures on click
def _grid(paths, width=300) -> str:
    imgs = "".join(f'<img src="{u}">' for u in (thumb(Path(x), width) for x in paths) if u)
    return f'<div class="pgrid">{imgs}</div>' if imgs else '<div class="quiet">None saved.</div>'


def _dialog(title):
    d = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
    return d(title, width="large") if d else (lambda fn: fn)


def person_pictures(kind: str, ident) -> tuple[list, list]:
    """(face pictures, body pictures) saved for one person, as absolute paths. Unknown people: only what the database lists."""
    faces, bodies = [], []
    if kind == "known":
        folder = DATA_DIR / "known" / str(ident)                  # data/known/<name>/01.jpg ..., body_01.jpg ...
        files = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
        faces = [str(f.resolve()) for f in files if not f.name.startswith("body")]
        bodies = [str(f.resolve()) for f in files if f.name.startswith("body")]
    else:
        for lst, tbl in ((faces, "unknown_embeddings"), (bodies, "unknown_body")):
            df = query(f"SELECT image_path FROM {tbl} WHERE unknown_id=? ORDER BY id", (int(ident),))
            for x in (df["image_path"].tolist() if not df.empty else []):
                q = resolve(x)
                if q is not None and str(q.resolve()) not in lst:
                    lst.append(str(q.resolve()))
    return faces, bodies


NAME_RE = r"[A-Za-z0-9 _.\-]{1,40}"


def _tracker_busy() -> bool:
    _s, run, _f = live_state()
    return owned_running() or run


def _say(kind: str, msg: str):
    st.session_state["dlg_msg"] = (kind, msg)


def _save_unknown_name(uid: int):
    name = str(st.session_state.get(f"uname_{uid}", "")).strip()
    if name and not re.fullmatch(NAME_RE, name):
        _say("error", "Use letters, numbers, spaces, dot, dash or underscore only.")
        return
    con = _connect(write=True)
    if con is None:
        _say("error", "Could not open the database for writing.")
        return
    try:
        con.execute("UPDATE unknown_persons SET name=? WHERE id=?", (name or None, uid))
        con.commit()
        _say("success", f"Saved the name '{name}'." if name else "Name removed.")
    except sqlite3.Error as e:
        _say("error", f"Could not save: {e}")
    finally:
        con.close()


def _delete_picture(kind: str, ident, path: str, group: str):
    if _tracker_busy():
        _say("error", "Stop the tracker first (it keeps faces in memory), then delete pictures.")
        return
    con = _connect(write=True)
    if con is None:
        _say("error", "Could not open the database for writing.")
        return
    try:
        if kind == "unknown":
            tbl = "unknown_embeddings" if group == "face" else "unknown_body"
            rows = con.execute(f"SELECT id, image_path FROM {tbl} WHERE unknown_id=?", (int(ident),)).fetchall()
            ids = [i for i, ip in rows if resolve(ip) is not None and str(resolve(ip).resolve()) == path]
            for i in ids:
                con.execute(f"DELETE FROM {tbl} WHERE id=?", (i,))
        else:
            pid = con.execute("SELECT id FROM known_persons WHERE name=?", (str(ident),)).fetchone()
            if not pid:
                _say("error", "Person not found.")
                return
            folder = DATA_DIR / "known" / str(ident)
            if group == "face":
                files = sorted(f for f in folder.glob("*.jpg") if not f.name.startswith("body"))
                emb = [r[0] for r in con.execute("SELECT id FROM known_embeddings WHERE person_id=? ORDER BY id", (pid[0],))]
                if len(files) != len(emb):
                    _say("error", "The saved pictures and face fingerprints of this person do not line up one to one, "
                                  "so a single picture cannot be removed safely. Delete the person and enrol again.")
                    return
                if len(files) <= 1:
                    _say("error", "This is the last face picture. Delete the person instead.")
                    return
                idx = [str(f.resolve()) for f in files].index(path)
                con.execute("DELETE FROM known_embeddings WHERE id=?", (emb[idx],))
            else:
                try:
                    for i, ip in con.execute("SELECT id, image_path FROM known_body WHERE person_id=?", (pid[0],)).fetchall():
                        q = resolve(ip)
                        if q is not None and str(q.resolve()) == path:
                            con.execute("DELETE FROM known_body WHERE id=?", (i,))
                except sqlite3.Error:
                    pass
        con.commit()
    except (sqlite3.Error, ValueError) as e:
        _say("error", f"Could not delete: {e}")
        return
    finally:
        con.close()
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
    _say("success", "Picture deleted. Start the tracker again to use the change.")


def _authorise_unknown(uid: int):
    """Turn an unknown person into an enrolled one: copy their saved faces (and body pictures) under a name, then remove
    the unknown record. Past alerts stay as they were. The change is written to data/audit.log."""
    name = str(st.session_state.get(f"auth_name_{uid}", "")).strip()
    if not PASSWORD:
        _say("error", "Set DASHBOARD_PASSWORD in .env to use this button.")
        return
    if _tracker_busy():
        _say("error", "Stop the tracker first, then authorise.")
        return
    if not re.fullmatch(NAME_RE, name):
        _say("error", "Enter a name (letters, numbers, spaces, dot, dash or underscore).")
        return
    con = _connect(write=True)
    if con is None:
        _say("error", "Could not open the database for writing.")
        return
    try:
        if con.execute("SELECT 1 FROM known_persons WHERE lower(name)=lower(?)", (name,)).fetchone():
            _say("error", f"'{name}' is already enrolled. Choose another name.")
            return
        femb = con.execute("SELECT emb, image_path FROM unknown_embeddings WHERE unknown_id=? ORDER BY id", (uid,)).fetchall()
        bemb = con.execute("SELECT emb, image_path FROM unknown_body WHERE unknown_id=? ORDER BY id", (uid,)).fetchall()
        if not femb:
            _say("error", "This person has no saved face, so they cannot be authorised (a face is needed). Enrol them with a camera.")
            return
        cur = con.execute("INSERT INTO known_persons(name, created) VALUES(?,?)", (name, time.time()))
        pid = cur.lastrowid
        folder = DATA_DIR / "known" / name
        folder.mkdir(parents=True, exist_ok=True)
        copied: dict = {}
        n_f = n_b = 0
        for emb, ip in femb:
            con.execute("INSERT INTO known_embeddings(person_id, emb) VALUES(?,?)", (pid, emb))
            q = resolve(ip)
            if q is not None and str(q.resolve()) not in copied:
                n_f += 1
                dst = folder / f"{n_f:02d}.jpg"
                shutil.copyfile(q, dst)
                copied[str(q.resolve())] = dst
        for emb, ip in bemb:
            q = resolve(ip)
            key = str(q.resolve()) if q is not None else None
            if key and key not in copied:
                n_b += 1
                dst = folder / f"body_{n_b:02d}.jpg"
                shutil.copyfile(q, dst)
                copied[key] = dst
            con.execute("INSERT INTO known_body(person_id, emb, image_path) VALUES(?,?,?)",
                        (pid, emb, str(copied.get(key, ""))))
        for t in ("unknown_embeddings", "unknown_body"):
            con.execute(f"DELETE FROM {t} WHERE unknown_id=?", (uid,))
        con.execute("DELETE FROM unknown_persons WHERE id=?", (uid,))
        con.commit()
    except (sqlite3.Error, OSError) as e:
        _say("error", f"Could not authorise: {e}")
        return
    finally:
        con.close()
    shutil.rmtree(DATA_DIR / "unknown" / f"unknown_{uid}", ignore_errors=True)
    try:
        with open(DATA_DIR / "audit.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  AUTHORISED unknown #{uid} as '{name}'\n")
    except OSError:
        pass
    st.session_state["person_msg"] = ("success", f"Unknown #{uid} is now authorised as {name}. Start the tracker again to use the change.")
    st.session_state["dlg_close"] = True


def show_person(kind: str, ident, title: str):
    @_dialog(title)
    def _body():
        faces, bodies = person_pictures(kind, ident)
        if kind == "unknown":
            row = query("SELECT name, times_seen, last_seen FROM unknown_persons WHERE id=?", (int(ident),))
            cur = (row["name"].iloc[0] or "") if not row.empty and "name" in row.columns else ""
            nc1, nc2 = st.columns([3, 1])
            nc1.text_input("Name (a label for this person, it does not change who is authorised)", value=cur,
                           key=f"uname_{ident}", max_chars=40, placeholder="e.g. Saroj, visitor")
            nc2.write("")
            nc2.write("")
            nc2.button("Save name", key=f"savename_{ident}", on_click=_save_unknown_name, args=(int(ident),), **STRETCH)
        kind_, msg_ = st.session_state.get("dlg_msg", (None, ""))
        if kind_ and msg_:
            getattr(st, kind_)(msg_)
        for group, paths in (("face", faces), ("body", bodies)):
            st.markdown(f"**{group.title()} pictures ({len(paths)})**")
            if not paths:
                st.markdown('<div class="quiet">None saved.</div>', unsafe_allow_html=True)
            for r0 in range(0, len(paths), 5):
                cols = st.columns(5)
                for j, pth in enumerate(paths[r0:r0 + 5]):
                    u = thumb(Path(pth), 240)
                    if u:
                        cols[j].markdown(f'<img class="dpic" src="{u}">', unsafe_allow_html=True)
                    cols[j].button("Delete", key=f"dp_{group}_{r0 + j}", on_click=_delete_picture,
                                   args=(kind, ident, pth, group))
        if kind == "unknown":
            st.markdown("---")
            st.markdown("**Authorise this person**")
            if not PASSWORD:
                st.caption("To turn an unknown person into an authorised one, set DASHBOARD_PASSWORD in .env first, "
                           "then restart the dashboard.")
            else:
                if len(faces) < 4:
                    st.warning(f"Only {len(faces)} face picture(s) are saved, which is weaker than a normal enrolment "
                               "(8 samples with head turns). For someone who should stay authorised, use "
                               "+ Enrol new person instead.")
                st.text_input("Name to enrol as", key=f"auth_name_{ident}", max_chars=40, placeholder="e.g. Saroj Kumar")
                ok = st.checkbox("I understand: this person will be allowed to take objects without an alert.",
                                 key=f"auth_ok_{ident}")
                st.button("Authorise", key=f"auth_go_{ident}", type="primary", disabled=not ok,
                          on_click=_authorise_unknown, args=(int(ident),))
                if st.session_state.pop("dlg_close", False):
                    st.rerun()                                  # closes the window; the page shows the result
    _body()


def _ask_delete_person(kind: str, ident):
    _s, run, _f = live_state()
    if owned_running() or run:
        st.session_state["person_msg"] = ("error", "Stop the tracker first (it keeps people in memory), then delete.")
        return
    st.session_state["person_msg"] = (None, "")
    st.session_state["person_del"] = (kind, ident)


def _delete_person(kind: str, ident) -> str:
    """Remove an enrolled person (by name) or an unknown person (by number): database rows and saved pictures.
    The alert log and its snapshots stay."""
    con = _connect(write=True)
    if con is None:
        return "Could not open the database for writing."
    try:
        if kind == "known":
            row = con.execute("SELECT id FROM known_persons WHERE name=?", (ident,)).fetchone()
            if row:
                for t in ("known_embeddings", "known_body"):
                    try:
                        con.execute(f"DELETE FROM {t} WHERE person_id=?", (row[0],))
                    except sqlite3.Error:
                        pass
                con.execute("DELETE FROM known_persons WHERE id=?", (row[0],))
            folder = DATA_DIR / "known" / str(ident)
        else:
            for t in ("unknown_embeddings", "unknown_body"):
                try:
                    con.execute(f"DELETE FROM {t} WHERE unknown_id=?", (int(ident),))
                except sqlite3.Error:
                    pass
            con.execute("DELETE FROM unknown_persons WHERE id=?", (int(ident),))
            folder = DATA_DIR / "unknown" / f"unknown_{int(ident)}"
        con.commit()
    finally:
        con.close()
    try:                                              # only ever inside data/known or data/unknown
        base = (DATA_DIR / ("known" if kind == "known" else "unknown")).resolve()
        if folder.is_dir() and base in folder.resolve().parents:
            shutil.rmtree(folder, ignore_errors=True)
    except OSError:
        pass
    return "Deleted."


def confirm_delete_person(kind: str, ident):
    label_ = str(ident) if kind == "known" else f"Unknown #{ident}"

    @_dialog("Delete " + label_)
    def _body():
        _s, run, _f = live_state()
        if owned_running() or run:
            st.error("Stop the tracker first (it keeps people in memory), then delete.")
            return
        st.warning(f"Delete {label_}? Their saved faces, body pictures and fingerprints are removed. "
                   "The pickup log and its snapshots stay.")
        y1, y2 = st.columns(2)
        if y1.button("Delete", key="person_del_yes", type="primary", **STRETCH):
            try:
                msg = _delete_person(kind, ident)
            except Exception as e:                                    # e.g. database locked by another window
                msg = f"Could not delete: {e}"
            if msg == "Deleted.":
                st.session_state["person_msg"] = ("success", f"{label_} deleted. Start the tracker again to use the change.")
                st.rerun()
            st.error(msg)
        if y2.button("Cancel", key="person_del_no", **STRETCH):
            st.rerun()
    _body()


def show_person_msg():
    kind, msg = st.session_state.pop("person_msg", (None, ""))
    if kind and msg:
        getattr(st, kind)(msg)


def person_tile(col, key: str, title: str, meta: str, faces: list, bodies: list, status: str, kind: str = "known", ident=None):
    """A card with ONE picture (the face; the body only if no face was ever saved). Clicking it opens every saved picture."""
    first = (faces or bodies or [None])[0]
    u = thumb(Path(first), 220) if first else None
    with col:
        with kcontainer("ptile_" + key):
            st.markdown(f'<div class="pcard"><div class="imgs">{f"<img src=\"{u}\">" if u else "<div class=ph></div>"}</div>'
                        f'<div class="n">{esc(title)}</div><div class="m">{esc(meta)}</div>'
                        f'<div class="m">{esc(status)} &middot; {len(faces)} face, {len(bodies)} body pictures</div></div>',
                        unsafe_allow_html=True)
            if st.button("Open " + title, key="open_" + key):
                st.session_state["dlg_msg"] = (None, "")
                show_person(kind, ident, title)
            if st.button("Delete " + title, key="camdel_p" + key):
                confirm_delete_person(kind, ident)


# ---------------------------------------------------------------- page: people
def page_people():
    topbar()
    show_person_msg()
    t1, t2 = st.tabs(["Authorized people", "Unauthorized people"])
    with t1:
        enrol_panel()
        kp = query("SELECT p.name AS Name, COUNT(e.id) AS Samples, p.created AS created FROM known_persons p "
                   "LEFT JOIN known_embeddings e ON e.person_id = p.id GROUP BY p.id ORDER BY p.name")
        if kp.empty:
            st.markdown('<div class="empty"><h3>Nobody is enrolled</h3>Add people with <code>python enroll.py --name NAME</code></div>', unsafe_allow_html=True)
        else:
            kp["Enrolled on"] = pd.to_datetime(kp["created"], unit="s").dt.strftime("%Y-%m-%d")
            cols = st.columns(3)
            for i, (_, r) in enumerate(kp.iterrows()):
                faces, bodies = person_pictures("known", str(r["Name"]))
                person_tile(cols[i % 3], f"k{i}", str(r["Name"]), f'{int(r["Samples"])} samples, enrolled {r["Enrolled on"]}',
                            faces, bodies, "Authorised", "known", str(r["Name"]))
            st.caption("Click a person to see every saved picture, delete single pictures, or (unauthorized people) name or authorise them.")
    with t2:
        up = query("SELECT id, first_seen, last_seen, times_seen, name FROM unknown_persons ORDER BY last_seen DESC")
        if up.empty:
            st.markdown('<div class="empty"><h3>No unauthorized people yet</h3>Strangers who take an object are saved here.</div>', unsafe_allow_html=True)
        else:
            cols = st.columns(3)
            for i, (_, r) in enumerate(up.iterrows()):
                uid = int(r["id"])
                fpaths, bpaths = person_pictures("unknown", uid)
                seen = datetime.fromtimestamp(r["last_seen"]).strftime("%Y-%m-%d %H:%M")
                nm = str(r["name"]).strip() if pd.notna(r["name"]) and str(r["name"]).strip() else ""
                person_tile(cols[i % 3], f"u{uid}", nm or f"Unknown #{uid}",
                            (f"Unknown #{uid}. " if nm else "") + f'Seen {int(r["times_seen"])} time(s), last {seen}',
                            fpaths, bpaths, "Unauthorised", "unknown", uid)


# ---------------------------------------------------------------- page: objects
def object_files(oid) -> list:
    folder = DATA_DIR / "objects" / str(oid)
    return sorted(str(f.resolve()) for f in folder.glob("*.jpg")) if folder.is_dir() else []


def _save_object_name(oid: int):
    nm = str(st.session_state.get(f"oname_{oid}", "")).strip()
    if not nm or not re.fullmatch(r"[A-Za-z0-9 _.\-]{1,40}", nm):
        st.session_state["obj_dlg_msg"] = ("error", "Enter a name: letters, numbers, spaces, dot, dash or underscore (up to 40).")
        return
    con = _connect(write=True)
    if con is None:
        st.session_state["obj_dlg_msg"] = ("error", "Could not open the database.")
        return
    try:
        con.execute("UPDATE objects SET name=? WHERE id=?", (nm, int(oid)))
        con.commit()
        st.session_state["obj_dlg_msg"] = ("success", f"Saved the name '{nm}'.")
    finally:
        con.close()


def _delete_object_picture(oid, path: str):
    if Path(path).name == "crop.jpg":
        st.session_state["obj_dlg_msg"] = ("error", "crop.jpg is the tile picture. Delete the whole object instead.")
        return
    try:
        Path(path).unlink(missing_ok=True)
        st.session_state["obj_dlg_msg"] = ("success", "Picture deleted.")
    except OSError as e:
        st.session_state["obj_dlg_msg"] = ("error", f"Could not delete: {e}")


def _ask_delete_object(oid):
    st.session_state["obj_msg"] = (None, "")
    st.session_state["obj_del"] = int(oid)


def _delete_object(oid: int):
    con = _connect(write=True)
    if con is not None:
        try:
            con.execute("DELETE FROM objects WHERE id=?", (int(oid),))
            con.commit()
        finally:
            con.close()
    shutil.rmtree(DATA_DIR / "objects" / str(int(oid)), ignore_errors=True)


def confirm_delete_object(oid: int, title: str):
    @_dialog("Delete " + str(title))
    def _body():
        st.warning("Delete this object? Its saved pictures are removed. The pickup log and its snapshots stay.")
        y1, y2 = st.columns(2)
        if y1.button("Delete", key="obj_del_yes", type="primary", **STRETCH):
            try:
                _delete_object(oid)
            except Exception as e:
                st.error(f"Could not delete: {e}")
            else:
                st.session_state["obj_msg"] = ("success", "Object deleted.")
                st.rerun()
        if y2.button("Cancel", key="obj_del_no", **STRETCH):
            st.rerun()
    _body()


def show_object(oid: int, title: str):
    @_dialog(title)
    def _body():
        row = query("SELECT label, name, camera, created, last_used, times_used FROM objects WHERE id=?", (int(oid),))
        if row.empty:
            st.caption("This object no longer exists.")
            return
        r = row.iloc[0]
        st.caption(f"{r['label']} - {r['camera']} - added {datetime.fromtimestamp(r['created']):%Y-%m-%d %H:%M}, "
                   f"used {int(r['times_used'] or 1)} time(s), last {datetime.fromtimestamp(r['last_used']):%Y-%m-%d %H:%M}")
        c1, c2 = st.columns([3, 1])
        c1.text_input("Name", value=str(r["name"] or r["label"]), key=f"oname_{oid}", max_chars=40, placeholder="e.g. Blue bottle")
        c2.write("")
        c2.write("")
        c2.button("Save name", key=f"osave_{oid}", on_click=_save_object_name, args=(int(oid),), **STRETCH)
        kind_, msg_ = st.session_state.get("obj_dlg_msg", (None, ""))
        if kind_ and msg_:
            getattr(st, kind_)(msg_)
        files = object_files(oid)
        st.markdown(f"**Pictures ({len(files)})**")
        st.caption("crop = the object, scene = the whole frame it was outlined on, run = the object each time it was reused.")
        for r0 in range(0, len(files), 5):
            cols = st.columns(5)
            for j, pth in enumerate(files[r0:r0 + 5]):
                u = thumb(Path(pth), 240)
                nm_ = Path(pth).stem
                if u:
                    cols[j].markdown(f'<img class="dpic" src="{u}"><div class="quiet" style="margin-top:2px">{esc(nm_)}</div>',
                                     unsafe_allow_html=True)
                if nm_ != "crop":
                    cols[j].button("Delete", key=f"odp_{r0 + j}", on_click=_delete_object_picture, args=(oid, pth))
        ev = load_events()
        mine = ev[ev["item"] == str(r["name"] or r["label"])] if not ev.empty else ev
        st.markdown(f"**Pickups ({len(mine)})**")
        if mine.empty:
            st.markdown('<div class="quiet">No pickups logged for this object yet.</div>', unsafe_allow_html=True)
        for _, e in mine.head(6).iterrows():
            headline, label_, kind = what_happened(e)
            a, b = st.columns([1, 4])
            u = thumb(resolve(e["snapshot"]), 160)
            if u:
                a.markdown(f'<img class="dpic" src="{u}">', unsafe_allow_html=True)
            b.markdown(f'**{esc(headline)}**  \n{esc(label_)} - {e["dt"]:%Y-%m-%d %H:%M}')
    _body()


def object_tile(col, oid: int, title: str, meta: str, status: str, first):
    u = thumb(Path(first), 220) if first else None
    with col:
        with kcontainer(f"ptile_o{oid}"):
            st.markdown(f'<div class="pcard"><div class="imgs">{f"<img src=\"{u}\">" if u else "<div class=ph></div>"}</div>'
                        f'<div class="n">{esc(title)}</div><div class="m">{esc(meta)}</div><div class="m">{esc(status)}</div></div>',
                        unsafe_allow_html=True)
            if st.button("Open " + title, key=f"open_o{oid}"):
                st.session_state["obj_dlg_msg"] = (None, "")
                show_object(oid, title)
            if st.button("Delete " + title, key=f"camdel_po{oid}"):
                confirm_delete_object(oid, title)


def page_objects():
    topbar()
    kind_, msg_ = st.session_state.pop("obj_msg", (None, ""))
    if kind_ and msg_:
        getattr(st, kind_)(msg_)
    objs = query("SELECT id, label, name, camera, created, last_used, times_used FROM objects ORDER BY last_used DESC")
    if objs.empty:
        st.markdown('<div class="empty"><h3>No objects yet</h3>Objects you outline when you press Start are saved here, '
                    'with their pictures.</div>', unsafe_allow_html=True)
        return
    ev = load_events()
    counts = ev["item"].value_counts().to_dict() if not ev.empty else {}
    cols = st.columns(3)
    for i, (_, r) in enumerate(objs.iterrows()):
        oid = int(r["id"])
        files = object_files(oid)
        first = next((f for f in files if Path(f).name == "crop.jpg"), files[0] if files else None)
        nm = str(r["name"] or r["label"])
        n = int(counts.get(nm, 0))
        object_tile(cols[i % 3], oid, nm,
                    f"{r['camera']}, added {datetime.fromtimestamp(r['created']):%Y-%m-%d}",
                    f"{n} pickup{'' if n == 1 else 's'} - used {int(r['times_used'] or 1)} time(s) - {len(files)} pictures", first)
    st.caption("Click an object to see its pictures and pickups, or rename it. Use the bin icon to remove it.")


{"dash": page_dash, "log": page_log, "objects": page_objects, "people": page_people}[st.session_state["page"]]()
