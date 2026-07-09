# Realtime Obstacle Detection (YOLOv8)

A real-time computer vision pipeline that detects and tracks obstacles with **YOLOv8 + ByteTrack**, estimates their **real-world distance** using a pinhole camera model, and turns that into a **stable navigation decision** (forward / left / right / backward) using temporal smoothing and hysteresis.

Built to explore the full perception-to-action loop: detection → tracking → distance estimation → danger scoring → smoothed decision-making — not just running a model on a webcam.

## Why this project

Most YOLO + webcam demos stop at "draw a box, print left/center/right." The interesting (and harder) part is what happens *after* detection: a single noisy frame can cause a false alarm, a bounding box sitting on a zone boundary can flicker between zones every frame, and box area alone is a poor proxy for how dangerous an obstacle actually is. This project works through those problems directly:

- **ByteTrack** gives each object a stable ID across frames instead of re-detecting from scratch every frame.
- **Pinhole-camera distance estimation** (`distance = real_height × focal_length / pixel_height`) converts bounding box size into an actual real-world distance, with a built-in calibration helper.
- **Soft zone boundaries** (smoothstep interpolation) replace hard left/center/right cutoffs, so an object near a zone edge doesn't flicker between zones.
- **Per-track EMA smoothing + zone-level smoothing + decision hysteresis** — three layers that together prevent the system from twitching left-right-left on a single bad frame, at the cost of a small (<200ms) reaction delay.

## Features

- Real-time YOLOv8 object detection with optional ByteTrack multi-object tracking
- Real-world distance estimation via pinhole camera model, with a one-time calibration mode
- Soft (smoothstep) navigation zones instead of hard cutoffs
- Multi-layer danger score smoothing (per-track EMA + zone-level temporal averaging)
- Decision hysteresis to prevent rapid direction flickering
- Live HUD: bounding boxes with track ID/distance, zone danger bars, decision arrow, FPS/inference stats, command history
- Works with a real robot over serial, **or** in simulation mode with just a webcam (no hardware required)
- Runs on CPU, CUDA, or Apple Silicon (MPS) automatically

## Demo

*(Add a screenshot or GIF of the HUD running here — this is the single highest-impact addition you can make to this README.)*

## Project structure

```
.
├── main.py        # Entry point: CLI args, camera loop, calibration mode
├── config.py       # All tunable parameters (camera, model, zones, smoothing, theme)
├── models.py       # Detection / FrameResult data structures
├── detector.py     # YOLOv8 + ByteTrack wrapper: zones, distance, danger scoring
├── robot.py        # Serial robot controller (or simulation mode)
├── decision.py     # Hysteresis-based decision engine
├── hud.py          # Visualization / on-screen HUD
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/realtime-obstacle-detection-yolov8.git
cd realtime-obstacle-detection-yolov8
pip install -r requirements.txt
```

## Usage

```bash
# Simulation mode: webcam only, no hardware needed
python main.py

# Real robot connected over serial
python main.py --port /dev/ttyUSB0

# More accurate (slower) model
python main.py --model yolov8s.pt

# Headless mode (no display window, logs to console)
python main.py --no-display

# One-time distance calibration (recommended before real use)
python main.py --calibrate
```

### Distance calibration

Bounding-box-based distance is only accurate once calibrated for your specific camera. To calibrate:

1. Place any object of known height (e.g. yourself, or a chair) at a known distance from the camera — e.g. exactly 2.0 metres.
2. Run `python main.py --calibrate` and press `c` when the object is detected.
3. Enter the object's real height and its distance from the camera when prompted.
4. The script prints a `FOCAL_LENGTH_PX` value — copy it into `config.py`.

Without calibration, distance values are still useful for *relative* "closer vs. farther" comparisons, just not accurate in exact metres.

### Keyboard controls (while running, display mode)

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `+` / `=` | Increase danger threshold |
| `-` | Decrease danger threshold |

## How it works

1. **Detect + track** — each frame is passed to YOLOv8 (optionally with ByteTrack) to get bounding boxes, class labels, confidences, and persistent track IDs.
2. **Estimate distance** — each box's pixel height is converted to a real-world distance using the pinhole camera model and per-class known heights (e.g. average person height).
3. **Score danger** — each detection gets a danger score combining confidence, relative box size, vertical position in frame, object class priority, and inverse distance (closer = more dangerous).
4. **Smooth** — danger scores are smoothed per-track (EMA) and then across zones (rolling average) to suppress single-frame noise.
5. **Assign to zones** — each detection contributes to left/right zone danger using soft (smoothstep) zone membership instead of a hard cutoff.
6. **Decide** — the decision engine compares left/right zone danger to a threshold and only changes the active decision after several consecutive frames agree (hysteresis), then sends the corresponding command to the robot (or simulates it).

## Requirements

- Python 3.9+
- A webcam (for live use) or video source
- Optional: a robot with a serial-connected microcontroller expecting single-byte commands (`F`/`B`/`L`/`R`/`S`)

## License

MIT — see [LICENSE](LICENSE).
