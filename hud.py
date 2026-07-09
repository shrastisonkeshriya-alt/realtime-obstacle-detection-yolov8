"""
HUD / visualizer: draws bounding boxes, zone panels, decision arrow,
and a live stats panel on top of the camera frame.
"""

import cv2
import numpy as np

from config import CFG
from models import Detection, FrameResult
from robot import RobotController


def _danger_color(danger: float) -> tuple:
    t = min(danger / CFG.DANGER_THRESHOLD, 1.0)
    if t < 0.5:
        r = int(30  + t * 2 * (255 - 30))
        g = int(200 - t * 2 * (200 - 165))
        b = 30
    else:
        t2 = (t - 0.5) * 2
        r  = 255
        g  = int(165 - t2 * 165)
        b  = 30
    return (b, g, r)


def _draw_box(vis, det: Detection, cfg: "CFG.__class__"):
    x1, y1, x2, y2 = det.box
    color = cfg.class_color(det.class_id)

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    cs = 10
    for (cx, cy, sx, sy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(vis, (cx, cy), (cx + sx * cs, cy), color, 3)
        cv2.line(vis, (cx, cy), (cx, cy + sy * cs), color, 3)

    id_txt = f"#{det.track_id} " if det.track_id is not None else ""
    dist_txt = f" {det.distance_m:.1f}m" if det.distance_m is not None else ""
    label_txt = f"{id_txt}{det.label} {det.confidence:.0%}{dist_txt}"
    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad = 4
    pill_y = max(y1 - th - pad * 2, 0)
    cv2.rectangle(vis, (x1 - 1, pill_y), (x1 + tw + pad * 2, pill_y + th + pad * 2),
                  color, -1)
    cv2.putText(vis, label_txt, (x1 + pad, pill_y + th + pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)

    if det.zone != "center":
        badge = f"! {det.danger:.2f}"
        cv2.putText(vis, badge, (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CFG.THEME["highlight"], 1, cv2.LINE_AA)


def _draw_zone_panel(vis, x1, x2, label, danger, h):
    color = _danger_color(danger)
    fill  = min(int((danger / CFG.DANGER_THRESHOLD) * h), h)

    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.rectangle(vis, (x1, 0), (x2, h), color, 2)

    bar_x = x2 - 8 if label == "LEFT" else x1 + 4
    bar_bg_rect = ((bar_x, 20), (bar_x + 4, h - 20))
    cv2.rectangle(vis, bar_bg_rect[0], bar_bg_rect[1], (40, 40, 40), -1)
    if fill > 0:
        cv2.rectangle(vis, (bar_x, h - 20 - fill + 40),
                      (bar_x + 4, h - 20), color, -1)

    pct = min(danger / CFG.DANGER_THRESHOLD * 100, 100)
    cv2.putText(vis, f"{label}", (x1 + 6, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CFG.THEME["text"], 1, cv2.LINE_AA)
    cv2.putText(vis, f"{pct:.0f}%", (x1 + 6, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_hud(vis, result: FrameResult, robot: RobotController):
    h, w = vis.shape[:2]
    le   = int(w * CFG.LEFT_ZONE_END)
    rs   = int(w * CFG.RIGHT_ZONE_START)

    _draw_zone_panel(vis, 0,  le, "LEFT",  result.left_danger,  h)
    _draw_zone_panel(vis, rs, w,  "RIGHT", result.right_danger, h)

    for lx in [le, rs]:
        cv2.line(vis, (lx, 0), (lx, h), CFG.THEME["accent"], 1)
        for dy in range(0, h, 20):
            cv2.line(vis, (lx, dy), (lx, dy + 10), CFG.THEME["accent"], 1)

    cx = (le + rs) // 2
    decision = result.decision
    arrow_color = CFG.THEME["safe"]    if "FORWARD"  in decision else \
                  CFG.THEME["danger"]  if "BACKWARD" in decision else \
                  CFG.THEME["warning"]
    arrows = {
        "FORWARD":  [(cx, h//2 + 40), (cx,      h//2 - 40)],
        "BACKWARD": [(cx, h//2 - 40), (cx,      h//2 + 40)],
        "RIGHT":    [(cx - 40, h//2), (cx + 40, h//2)],
        "LEFT":     [(cx + 40, h//2), (cx - 40, h//2)],
    }
    for key, pts in arrows.items():
        if key in decision:
            cv2.arrowedLine(vis, pts[0], pts[1], arrow_color, 3,
                            tipLength=0.35, line_type=cv2.LINE_AA)
            break

    banner_h = 48
    cv2.rectangle(vis, (0, h - banner_h), (w, h), (8, 10, 16), -1)
    cv2.line(vis, (0, h - banner_h), (w, h - banner_h), arrow_color, 2)

    (tw, _), _ = cv2.getTextSize(decision, cv2.FONT_HERSHEY_DUPLEX, 0.72, 1)
    cv2.putText(vis, decision, ((w - tw) // 2, h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, arrow_color, 1, cv2.LINE_AA)

    panel_w, panel_h = 190, 90
    panel_x = w - panel_w - 8
    overlay2 = vis.copy()
    cv2.rectangle(overlay2, (panel_x, 8), (w - 8, 8 + panel_h),
                  CFG.THEME["panel"], -1)
    cv2.addWeighted(overlay2, 0.8, vis, 0.2, 0, vis)
    cv2.rectangle(vis, (panel_x, 8), (w - 8, 8 + panel_h), CFG.THEME["accent"], 1)

    stats = [
        f"FPS  {result.fps:5.1f}",
        f"INF  {result.inference_ms:4.0f} ms",
        f"OBJ  {len(result.detections):3d}",
        f"THR  {CFG.DANGER_THRESHOLD:.2f}",
    ]
    for i, s in enumerate(stats):
        cv2.putText(vis, s, (panel_x + 8, 28 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CFG.THEME["text"], 1, cv2.LINE_AA)

    for i, (ts, cmd) in enumerate(reversed(robot.cmd_log)):
        alpha = 1.0 - i * 0.15
        colour = tuple(int(c * alpha) for c in CFG.THEME["subtext"])
        cv2.putText(vis, f"{ts}  {cmd}", (8, h - banner_h - 12 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    return vis


def draw_frame(frame: np.ndarray, result: FrameResult, robot: RobotController):
    vis = frame.copy()
    for det in result.detections:
        _draw_box(vis, det, CFG)
    vis = _draw_hud(vis, result, robot)
    return vis
