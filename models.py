"""
Shared data structures used across the pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    class_id:   int
    label:      str
    confidence: float
    box:        tuple
    zone:       str            # dominant zone, for drawing/labels only
    danger:     float          # smoothed, tracked danger score
    track_id:   Optional[int]
    distance_m: Optional[float]


@dataclass
class FrameResult:
    detections:     list
    left_danger:    float
    right_danger:   float
    left_blocked:   bool
    right_blocked:  bool
    decision:       str
    fps:            float
    inference_ms:   float
