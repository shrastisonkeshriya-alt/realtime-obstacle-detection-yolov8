"""
Configuration for the YOLO Obstacle Avoidance pipeline.

All tunable parameters live here: camera settings, model settings,
zone/danger thresholds, smoothing behavior, and visual theme.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Camera ────────────────────────────────────────────────────────────────
    CAMERA_INDEX:    int   = 0
    FRAME_WIDTH:     int   = 640
    FRAME_HEIGHT:    int   = 480
    FPS:             int   = 30

    # ── YOLO Detection ────────────────────────────────────────────────────────
    MODEL_PATH:      str   = "yolov8n.pt"     # s = accuracy/speed sweet spot on GPU
    CONF_THRESHOLD:  float = 0.30
    IOU_THRESHOLD:   float = 0.45
    IMG_SIZE:        int   = 512
    DEVICE:          str   = (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )           # "cpu", "cuda", "mps"
    HALF_PRECISION:  bool  = False             # FP16 inference on CUDA (free speed)

    # ── Tracking (ByteTrack, built into ultralytics) ──────────────────────────
    ENABLE_TRACKING: bool  = True
    TRACKER_CFG:     str   = "bytetrack.yaml"
    TRACK_HISTORY:   int   = 5                # frames of danger history per object

    # ── Distance estimation (pinhole camera model) ────────────────────────────
    # distance_m = (REAL_HEIGHT_M[class] * FOCAL_LENGTH_PX) / bbox_height_px
    # FOCAL_LENGTH_PX default below is a rough placeholder for a 640x480 webcam.
    # RUN --calibrate ONCE with a known object+distance to set this correctly.
    FOCAL_LENGTH_PX: float = 600.0
    REAL_HEIGHT_M: dict = field(default_factory=lambda: {
        "person": 1.65, "chair": 0.90, "couch": 0.85, "dining table": 0.75,
        "car": 1.50, "truck": 3.00, "bus": 3.20, "motorcycle": 1.20,
        "bicycle": 1.10, "dog": 0.50, "cat": 0.30,
    })
    DEFAULT_HEIGHT_M: float = 1.0            # fallback for unlisted classes
    DANGER_DISTANCE_M: float = 1.2           # objects closer than this = high danger

    # ── Navigation Zones (fraction of frame width) ────────────────────────────
    LEFT_ZONE_END:      float = 0.33
    RIGHT_ZONE_START:   float = 0.67
    ZONE_SOFTNESS:       float = 0.06        # how wide the soft transition band is

    # ── Danger scoring ────────────────────────────────────────────────────────
    DANGER_THRESHOLD:   float = 0.08
    PROXIMITY_WEIGHT:   float = 2.0
    HIGH_PRIORITY_CLASSES: tuple = (
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "chair", "dining table", "couch",
    )

    # ── Temporal smoothing / hysteresis ───────────────────────────────────────
    SMOOTH_FRAMES:     int   = 3
    HYSTERESIS_FRAMES:  int   = 4             # consecutive frames needed to switch decision
    EMA_ALPHA:          float = 0.4           # per-track exponential smoothing factor

    # ── Robot serial commands ─────────────────────────────────────────────────
    CMD_FORWARD:     bytes = b'F'
    CMD_BACKWARD:    bytes = b'B'
    CMD_LEFT:        bytes = b'L'
    CMD_RIGHT:       bytes = b'R'
    CMD_STOP:        bytes = b'S'
    BAUD_RATE:       int   = 9600
    SERIAL_TIMEOUT:  float = 1.0
    COMMAND_COOLDOWN:float = 0.25

    # ── Visual theme ──────────────────────────────────────────────────────────
    THEME = {
        "bg":        (10,  12,  18),
        "accent":    (0,   210, 255),
        "danger":    (30,  30,  220),
        "safe":      (30,  200, 100),
        "warning":   (30,  165, 255),
        "text":      (230, 235, 245),
        "subtext":   (120, 130, 145),
        "panel":     (20,  24,  35),
        "highlight": (255, 220, 60),
    }

    PALETTE: list = field(default_factory=lambda: [
        (255, 80,  80),  (80,  255, 80),  (80,  80,  255), (255, 255, 80),
        (255, 80,  255), (80,  255, 255), (200, 100, 50),   (50,  200, 100),
        (100, 50,  200), (200, 200, 50),  (50,  200, 200),  (200, 50,  200),
    ])

    def class_color(self, class_id: int) -> tuple:
        return self.PALETTE[class_id % len(self.PALETTE)]


CFG = Config()
