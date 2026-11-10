# Live Tracking System

Watches an IP camera, **follows the objects you select wherever they go**, finds out **who is holding them**, and checks that person's face: an enrolled person is **AUTHORISED**; anyone else raises an **ALERT**, their face is saved as "unknown #N" and recognised next time; if no usable face was seen the alert says **UNIDENTIFIED** (it never silently passes). This is a proof of concept (POC).

### Using the same objects again
Tick **Use the same objects as last time** on the Dashboard View page (or run `python main.py --reuse-zones`). The saved outline only says
where the object was, so the tracker first searches the **whole frame** for each object's saved picture (several sizes, upright or on its
side) and moves the outline to where it is now. An object that is still in place stays put; one that cannot be found keeps its saved place
and a message is printed in the tracker log.

### Body pictures are learned automatically
You can enrol a person with the face only. Whenever the tracker recognises an enrolled person by face and they are alone in their box,
it saves their body fingerprint and picture (up to `learn_body_max`, default 6, only when it is a clearly new view). The body never authorises
anyone, it only helps the record and re-identification. Settings under `faces:` in `config.yaml`: `learn_body` (true/false), `learn_body_max`.

## What you see and what is printed

| Event | Screen | Console |
|---|---|---|
| Object lying still | cyan box, `object 1: at rest` | - |
| Someone picks it up | red box, `object 1: with person #3 (Priya)` + a trail line showing where it went | `HOLDER: object 1 is now with person #3`, then `TAKEN: ...` |
| Enrolled person | green banner `AUTHORISED: object 1 taken by Priya` | same, plus snapshot path |
| Someone else | red banner `ALERT: object 1 taken by UNKNOWN person #2` + beep | same; face crop saved in `data\unknown\unknown_2\` |
| Handed to another person | box label changes | `HOLDER: ...`, `TAKEN: ... (handover)` and the check runs again |
| Put down somewhere new | back to cyan | `PUT DOWN: object 1 put down, now watched at its new place` |
| Hidden / out of view | grey box `searching...` | `LOST: object 1 is not visible (last seen with person #3)` after about 30 frames |

Events and snapshots are also stored in `data\` (see Privacy).

## Files

| File | What it does |
|---|---|
| `main.py` | Entry point: reads the camera, runs everything, draws the window. |
| `enroll.py` | Add, list or delete the allowed people. |
| `config.yaml` | All settings (camera, mode, thresholds). Comments explain each one. |
| `src/frame_source.py` | One interface for IP camera (RTSP), webcam and video files; keeps only the newest frame and reconnects when the stream drops. |
| `src/detector.py` | YOLO + tracker wrapper; the pose model gives each person's id, box and wrists. |
| `src/zones.py` | Start-up selection: live preview, freeze a frame, click a polygon outline around each object; saves `zones.json`. |
| `src/follow.py` | **Follow mode:** visual tracker (VitTrack) per selected object, with whole-frame re-search when the object is lost. |
| `src/pickup.py` | Decides pickup / hand-over / put-down from the object box and each person's wrists. |
| `src/watch.py` | `select` mode: watches the fixed spot where an object was selected. |
| `src/faces.py` | Face detection (YuNet) + embedding (SFace) with OpenCV's built-in models. |
| `src/body.py` | Body appearance fingerprint (person re-identification, 107 MB model). Only used to recognise a stranger again when the face is hidden. |
| `src/facedb.py` | SQLite: known people, unknown people (face + body), event log. |
| `src/authz.py` | Authorised / unauthorised / unidentified decision, snapshots, alert de-duplication. |
| `requirements.txt`, `.env.example`, `.gitignore` | Packages; camera address template; keeps secrets, models, footage and face data out of git. |

## 1. Install (Windows PowerShell, once)

```powershell
cd "D:\priya\live tracking system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first runs download small models (YOLO, face models about 39 MB, tracker about 0.7 MB), so internet is needed once.

## 2. Connect the IP camera

1. Find the camera's RTSP address (see `.env.example` for common formats) and make sure the PC can reach the camera (same network or VPN).
2. Check it in VLC first (Media > Open Network Stream), or run:
   ```powershell
   Copy-Item .env.example .env
   notepad .env        # put your camera's address in RTSP_URL, save
   python -c "import cv2,os; from dotenv import load_dotenv; load_dotenv(); c=cv2.VideoCapture(os.getenv('RTSP_URL')); print('opened:', c.isOpened())"
   ```
3. `config.yaml` already has `source: type: rtsp`.

Camera placement: fix the camera (it must not move after you select objects), at about head height or a little above, looking at where people stand to take the objects. Faces should be at least about 100 px wide in the image, so a wide ceiling camera far away will not recognise faces. If CPU is slow, start with the camera's sub stream (lower resolution) to test, then use the main stream.

## 3. Enrol the allowed people (once per person)

```powershell
python enroll.py --name Priya
```

Stand where people normally stand in front of the camera and slowly turn your head until 8 samples are taken. Run it again at another distance or lighting to add more samples. Other commands: `--images "D:\photos\priya"` (enrol from clear photos), `--list`, `--delete Priya`.

## 4. Run

```powershell
python main.py
```

1. A live preview opens. Put the objects where they normally lie and press **SPACE** to freeze a frame.
2. Outline the first object with a polygon: **left-click** each corner around it, then press **ENTER** (or right-click) to close the shape. **Backspace** undoes the last corner, **Esc** clears the current outline. Repeat for more objects. Press **C** when done. The outlines are saved in `zones.json`.
3. Following starts. Press **q** or **Esc** in the window to quit.

Next time, `python main.py --reuse-zones` skips the selection (camera must not have moved and the objects must be in place).

Other options: `--mode select` (only watch the selected spot), `--mode detect` (auto-track phones with YOLO, no selection), `--no-faces` (no authorisation check), `--source file --file clip.mp4` (test on a recording), `--device cuda:0` (GPU server).

## Dashboard (for monitoring and client demos)

A web page in the same light theme as the Live CCTV analytics dashboard: live picture, today's numbers, recent pickups with snapshots, an alert log with filters and CSV download, and the enrolled and unknown people.

```powershell
pip install -r requirements.txt            # adds streamlit and pandas, once
streamlit run dashboard/app.py             # opens in the browser at http://localhost:8502
```

Everything is done from the page, no second terminal needed. In **Camera and control** (Dashboard View, right side): **+ Add camera** (IP camera with address, port, user, password and stream, or a webcam number; **Test** checks it), then **Select**, then **Start**. The window to outline the objects opens on the PC; after that tracking runs and the live picture appears. **Stop** ends it. Cameras are saved in `data/cameras.json` (this file may contain the camera password and stays on the PC). You can still run `python main.py --source rtsp` in a terminal; the dashboard then shows it, but it must be stopped in that terminal.

- The dashboard reads what the tracker writes: `data/faces.db`, `data/snapshots` and two small files in `data/live` (the latest picture and a status file, written about every half second). It can be opened or closed at any time without affecting tracking.
- **People page, Authorized people:** press **+ Enrol new person**, type the name and press Start enrolment (stop the tracker first). A window opens on the PC using the selected camera: look at it, turn your head slowly and step back so your whole body is visible. Face samples and body fingerprints are saved; the face alone decides who is authorised.  The terminal commands still work: `python enroll.py --name NAME` and `python enroll.py --name NAME --body`.
- **Objects page:** every object you outline at Start is saved with a cut-out, the frame it was outlined on, the camera and the date (`data/objects/`). Click an object to rename it, see its pictures and pickups, or delete a picture; the bin icon removes the object. The pickup log keeps the tracker's label (`object 5`); the dashboard shows the name you gave it.
- **People page:** tabs *Authorized people* and *Unauthorized people*. On an unauthorized person you can add a name label, delete single pictures, or (with `DASHBOARD_PASSWORD` set) authorise them; authorising is written to `data/audit.log`.
- **Port:** the dashboard uses port 8502 (set in `.streamlit/config.toml`, so it does not clash with another Streamlit app on 8501). To use another one: edit `port` there, or run `streamlit run dashboard/app.py --server.port 8503`.
- Pages: **Dashboard View** (live view, five numbers, recent pickups, pickups per hour), **Alert log** (filters, table, snapshot viewer, CSV), **People** (Authorized people and Unauthorized people as picture tiles; click a tile to see every saved picture, delete single pictures, give an unauthorized person a name, or authorise them; a bin icon deletes the whole person).
- To ask for a password, add `DASHBOARD_PASSWORD=yourpassword` to `.env`. The pages show faces, so set one before showing it on a shared network.
- The title and subtitle can be changed under a `dashboard:` section in `config.yaml` (`title`, `subtitle`). `live_feed: false` stops the tracker writing the live picture.
- Colours come from `.streamlit/config.toml` (teal accent, navy sidebar).

## How follow mode works

1. At selection, a tracker is started on the outline's bounding rectangle (trackers need a rectangle) and a picture of the object is saved with the background outside the polygon masked out. The polygon is redrawn on screen following the object.
2. Every processed frame the tracker gives the object's new box and a confidence score. The pose model gives every person's id and wrist positions.
3. `pickup.py` treats the object box like a detection: a **pickup** is when a person's wrist stays near the object for a few frames and the object has moved away from its resting spot; the object is **held** by the person whose wrist is closest; if someone else becomes the closest for a few frames it is a **hand-over**; when no hand is near for a few frames it is **put down** and the new spot becomes its resting place.
4. On every pickup or hand-over, `authz.py` looks at that person's best face pictures (collected before the pickup too) and decides authorised / unauthorised / unidentified.
5. **Tracking never stops.** If the tracker loses the object (hidden in a hand, or its holder walked out of view), the system keeps searching the **whole frame** every few frames: anything that is new compared with the empty scene (and, for common things like bottles or phones, the detector's boxes) is compared with the object's size, shape and colours. A match that stays in the same place for a few looks is accepted, the tracker restarts there and that spot becomes the object's new resting place. This works while the holder is still in view (object put down next to them), after the holder has left, and when the object reappears much later.
6. **One name per person.** On screen, in the alert banner, in the log and in the dashboard a person is shown as their enrolled name or as **Unknown person #N** (the number of the saved stranger). Until the face has been checked the label says "person (identifying...)". The tracker's internal person numbers are no longer shown.

## Face and body of strangers

- **Authorisation uses the face only.** A body fingerprint depends on clothes and two people can dress alike, so it can never make someone AUTHORISED.
- When a stranger takes the object, the system saves their **face fingerprint and a body fingerprint** together as one "unknown #N".
- If the same stranger returns with the **face hidden**, the body fingerprint is matched against the saved strangers: "unknown #N again (recognised by body)". A new stranger whose face is hidden is saved by body only; when their face is seen later it is added to the same record.
- If neither a usable face nor a usable body picture was captured, the alert says UNIDENTIFIED.
- Body matches are only reliable on the same day / same clothes. The first run downloads a ~107 MB model. Switch it off with `faces: body_enabled: false`. Tune the match with `body_thresh` (0.75 is a starting value: raise it if different strangers are merged into one number, lower it if the same stranger gets new numbers).

## Tuning (in `config.yaml`)

| Problem | Change |
|---|---|
| Pickup not detected | raise `pickup: hand_margin` (1.2), or set `require_movement: false` while testing |
| False pickups when someone walks past | raise `hold_frames` (6) or `move_thresh` (1.2) |
| Object keeps getting lost | lower `follow: min_score` (0.25) or `reacquire_thresh` (0.6); choose a more distinctive object/box |
| Object put down somewhere new is not found again | lower `follow: novelty_diff` (12) or `anywhere_confirm` (2); keep the camera still and the lighting steady |
| Wrong thing picked up as the object | raise `follow: anywhere_confirm` (5) or `novelty_diff` (25) |
| Known person flagged as unknown | enrol more samples at that distance/lighting, lower `faces: known_thresh` (0.35) |
| Stranger accepted as known | raise `known_thresh` (0.5) |
| Always UNIDENTIFIED | faces too small or hidden: move the camera closer/lower, lower `min_face_px` (40), raise `face_wait_sec` |
| Same stranger gets new numbers, or different strangers get one number | `faces: body_thresh` (lower = merge more, higher = merge less) |
| Slow (low FPS) | raise `runtime: frame_skip` (3), use a lower camera resolution, or use the GPU server (`--device cuda:0`) |

## Known limitations

- **Fixed camera only.** If the camera moves, select the objects again.
- **Tracking is visual:** it can drift or lose the object when it is hidden, spins fast, is tiny, or looks like its surroundings. A distinctive object photographed clearly works best. It is not guaranteed; the program reports `LOST` instead of guessing.
- **Pickup needs the person's wrist visible.** Hands hidden behind the body or out of frame will not trigger it.
- The face check needs a usable face (about 60 px wide or more, front-ish). Without one, the result is `UNIDENTIFIED`, an alert, not a pass.
- The face recogniser (SFace) is lighter than ArcFace/InsightFace; use a stronger one for production.
- Body re-identification is only used to recognise strangers again (same clothes); it is not tuned on your camera yet and works best for people at least about 140 px tall in the picture.
- Not in this POC: Telegram/e-mail alerts, dashboard, multiple cameras.

Troubleshooting: `Could not open video source` means the RTSP address, password (URL-encode special characters) or network is wrong; the selection window does not respond: click it once, click the corners of the outline, then press ENTER; model download fails: put the `.onnx` files (OpenCV Zoo: YuNet, SFace, VitTrack) into `models\`.

## Privacy

`data\` and `models\` hold face and body images and biometric embeddings; they are git-ignored. Restrict access to that folder, tell people they are being recorded and delete data you no longer need. For India, check consent and retention rules (DPDP Act) before a real deployment.

### Authorising an unauthorized person from the dashboard

Open the person's tile under Unauthorized people, then use **Authorise this person**. It needs `DASHBOARD_PASSWORD` in `.env`, the tracker stopped, a name, and a tick in the confirmation box. Their saved faces and body pictures are copied under that name and the unknown record is removed. Past alerts stay as they were, and each authorisation is written to `data/audit.log`. A person authorised this way has only the few face samples saved at the time of the pickups; for someone who should stay authorised, use **+ Enrol new person** instead.
