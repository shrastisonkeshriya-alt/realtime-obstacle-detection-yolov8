"""
YOLOv8 + ByteTrack detector: zone-aware, distance-aware, danger-scored.
"""

import sys
import time
import collections
from typing import Optional

import numpy as np

from config import CFG
from models import Detection

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Run: pip install ultralytics")


class YOLODetector:
    """YOLOv8 + ByteTrack wrapper: zone-aware, distance-aware, danger-scored."""

    def __init__(self, model_path: str, device: str):
        if not YOLO_AVAILABLE:
            print("[ERROR] Install ultralytics:  pip install ultralytics")
            sys.exit(1)

        print(f"[YOLO] Loading {model_path} on {device} …")
        self.model  = YOLO(model_path)
        self.device = device
        self.names  = self.model.names
        self.frame_w = CFG.FRAME_WIDTH
        self.frame_h = CFG.FRAME_HEIGHT

        # Smoothed zone-level danger (for decision engine)
        self._left_hist  = collections.deque(maxlen=CFG.SMOOTH_FRAMES)
        self._right_hist = collections.deque(maxlen=CFG.SMOOTH_FRAMES)

        # Per-track EMA danger score: track_id -> float
        self._track_danger_ema: dict = {}
        # Track ids seen this frame (to decay/drop stale ones)
        self._seen_this_frame: set = set()

        print(f"[YOLO] Ready — {len(self.names)} COCO classes | "
              f"tracking={'ON' if CFG.ENABLE_TRACKING else 'OFF'}")

    # ── Soft zone membership (replaces hard left/center/right cutoff) ─────────
    def _zone_weights(self, cx_frac: float) -> tuple:
        """
        Returns (left_weight, right_weight) in [0,1], smoothly transitioning
        across the zone boundaries instead of flipping abruptly. This is what
        fixes the 'object sitting right on the 33% line flickers zones' bug.
        """
        s = CFG.ZONE_SOFTNESS

        def smoothstep(edge_lo, edge_hi, x):
            if edge_hi <= edge_lo:
                return 1.0 if x >= edge_lo else 0.0
            t = np.clip((x - edge_lo) / (edge_hi - edge_lo), 0.0, 1.0)
            return float(t * t * (3 - 2 * t))

        # left_weight = 1 near x=0, fades to 0 by LEFT_ZONE_END (+ softness band)
        left_weight = 1.0 - smoothstep(CFG.LEFT_ZONE_END - s, CFG.LEFT_ZONE_END + s, cx_frac)
        # right_weight = 0 until RIGHT_ZONE_START (- softness band), then rises to 1
        right_weight = smoothstep(CFG.RIGHT_ZONE_START - s, CFG.RIGHT_ZONE_START + s, cx_frac)
        return left_weight, right_weight

    def _estimate_distance(self, label: str, box_h_px: int) -> Optional[float]:
        if box_h_px <= 0:
            return None
        real_h = CFG.REAL_HEIGHT_M.get(label, CFG.DEFAULT_HEIGHT_M)
        return (real_h * CFG.FOCAL_LENGTH_PX) / box_h_px

    def _danger_score(self, confidence: float, box: tuple, distance_m: Optional[float],
                       label: str, frame_w: int, frame_h: int) -> float:
        """
        Danger combines: confidence, box-area/zone-area, vertical proximity,
        class priority, AND real-world distance (closer = much more dangerous).
        Distance is the most reliable signal, so it gets the dominant weight.
        """
        x1, y1, x2, y2 = box
        box_area  = max((x2 - x1) * (y2 - y1), 1)
        zone_w    = int(frame_w * CFG.LEFT_ZONE_END)
        zone_area = zone_w * frame_h

        proximity_pixel = CFG.PROXIMITY_WEIGHT if y2 > frame_h * 0.5 else 1.0
        boost = 1.5 if label in CFG.HIGH_PRIORITY_CLASSES else 1.0

        base = confidence * (box_area / zone_area) * proximity_pixel * boost

        if distance_m is not None:
            # Inverse-distance multiplier, capped so very close objects dominate
            # but we don't get divide-by-near-zero explosions.
            dist_factor = CFG.DANGER_DISTANCE_M / max(distance_m, 0.3)
            dist_factor = min(dist_factor, 4.0)
            return base * dist_factor
        return base

    def detect(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        predict_kwargs = dict(
            conf=CFG.CONF_THRESHOLD,
            iou=CFG.IOU_THRESHOLD,
            imgsz=CFG.IMG_SIZE,
            device=self.device,
            half=False,
            verbose=False,
        )

        if CFG.ENABLE_TRACKING:
            results = self.model.track(frame, persist=True, tracker=CFG.TRACKER_CFG,
                                        **predict_kwargs)
        else:
            results = self.model.predict(frame, **predict_kwargs)

        inference_ms = (time.perf_counter() - t0) * 1000

        detections   = []
        left_danger  = 0.0
        right_danger = 0.0
        self._seen_this_frame = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = CFG.ENABLE_TRACKING and boxes.id is not None

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf   = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                label  = self.names.get(cls_id, f"cls{cls_id}")
                track_id = int(boxes.id[i]) if has_ids else None

                cx_frac = ((x1 + x2) / 2) / w
                left_w, right_w = self._zone_weights(cx_frac)

                # For labeling/drawing only, pick the dominant zone
                if left_w > right_w and left_w > 0.1:
                    zone = "left"
                elif right_w > left_w and right_w > 0.1:
                    zone = "right"
                else:
                    zone = "center"

                distance_m = self._estimate_distance(label, y2 - y1)
                raw_danger = self._danger_score(conf, (x1, y1, x2, y2), distance_m,
                                                 label, w, h)

                # Per-track EMA smoothing so a single noisy frame can't cause a spike
                if track_id is not None:
                    self._seen_this_frame.add(track_id)
                    prev = self._track_danger_ema.get(track_id, raw_danger)
                    smoothed_danger = CFG.EMA_ALPHA * raw_danger + (1 - CFG.EMA_ALPHA) * prev
                    self._track_danger_ema[track_id] = smoothed_danger
                else:
                    smoothed_danger = raw_danger

                det = Detection(cls_id, label, conf, (x1, y1, x2, y2), zone,
                                 smoothed_danger, track_id, distance_m)
                detections.append(det)

                # Contribute to zone danger weighted by soft zone membership
                left_danger  = max(left_danger,  smoothed_danger * left_w)
                right_danger = max(right_danger, smoothed_danger * right_w)

            # Drop EMA memory for tracks that disappeared (avoid stale ghosts)
            stale = set(self._track_danger_ema) - self._seen_this_frame
            for tid in stale:
                del self._track_danger_ema[tid]

        # Zone-level temporal smoothing (kept from v2, on top of per-track EMA)
        self._left_hist.append(left_danger)
        self._right_hist.append(right_danger)
        left_smooth  = float(np.mean(self._left_hist))
        right_smooth = float(np.mean(self._right_hist))

        return (detections, left_smooth, right_smooth, inference_ms)
