"""
YOLO-Powered Robot Obstacle Avoidance — entry point.

Usage:
    python main.py                        # Simulation, webcam, GPU/MPS/CPU
    python main.py --port /dev/ttyUSB0    # Real robot over serial
    python main.py --model yolov8s.pt     # more accurate, slower
    python main.py --no-display           # Headless mode
    python main.py --calibrate            # One-time distance calibration helper

ONE-TIME CALIBRATION (do this for correct real-world distances):
    Place any object of KNOWN real height (e.g. a chair, or yourself) at a
    KNOWN distance from the camera (e.g. exactly 2.0 metres). Run:
        python main.py --calibrate
    It will print the pixel height of the detected box. Then set:
        FOCAL_LENGTH_PX = (pixel_height * known_distance_m) / real_object_height_m
    in config.py. Without calibration, distances are still USABLE for
    relative "closer vs farther" comparisons, just not exact in metres.
"""

import sys
import time
import argparse

import cv2

from config import CFG
from detector import YOLODetector
from robot import RobotController
from decision import DecisionEngine
from hud import draw_frame, _draw_box
from models import FrameResult


def run_calibration(detector: YOLODetector, cap: cv2.VideoCapture):
    print("\n[CALIBRATE] Place a known object at a known distance from the camera.")
    print("[CALIBRATE] Press 'c' to capture a reading, 'q' to quit calibration.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        detections, _, _, _ = detector.detect(frame)
        vis = frame.copy()
        for det in detections:
            _draw_box(vis, det, CFG)
        cv2.imshow("Calibration - press C to capture, Q to quit", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and detections:
            d = max(detections, key=lambda x: (x.box[3]-x.box[1])*(x.box[2]-x.box[0]))
            box_h = d.box[3] - d.box[1]
            print(f"[CALIBRATE] Detected '{d.label}' with pixel height = {box_h}px")
            try:
                real_h = float(input("  Enter its REAL height in metres: "))
                dist   = float(input("  Enter its distance from camera in metres: "))
                focal  = (box_h * dist) / real_h
                print(f"\n  >>> Set FOCAL_LENGTH_PX = {focal:.1f} in config.py <<<\n")
            except ValueError:
                print("  Invalid input, try again.")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Powered Robot Obstacle Avoidance (tracking + distance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port",       type=str,   default=None)
    parser.add_argument("--model",      type=str,   default=CFG.MODEL_PATH)
    parser.add_argument("--conf",       type=float, default=CFG.CONF_THRESHOLD)
    parser.add_argument("--iou",        type=float, default=CFG.IOU_THRESHOLD)
    parser.add_argument("--danger",     type=float, default=CFG.DANGER_THRESHOLD)
    parser.add_argument("--device",     type=str,   default=CFG.DEVICE)
    parser.add_argument("--camera",     type=int,   default=CFG.CAMERA_INDEX)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-tracking", action="store_true", help="Disable ByteTrack, fall back to per-frame detection")
    parser.add_argument("--img-size",   type=int,   default=CFG.IMG_SIZE)
    parser.add_argument("--calibrate",  action="store_true", help="Run distance calibration helper and exit")
    args = parser.parse_args()

    CFG.MODEL_PATH       = args.model
    CFG.CONF_THRESHOLD   = args.conf
    CFG.IOU_THRESHOLD    = args.iou
    CFG.DANGER_THRESHOLD = args.danger
    CFG.DEVICE           = args.device
    CFG.IMG_SIZE         = args.img_size
    CFG.ENABLE_TRACKING  = not args.no_tracking

    detector = YOLODetector(CFG.MODEL_PATH, CFG.DEVICE)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CFG.FPS)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)

    if args.calibrate:
        run_calibration(detector, cap)
        cap.release()
        return

    robot    = RobotController(port=args.port)
    engine   = DecisionEngine()

    print("\n" + "=" * 60)
    print("  YOLO Robot Obstacle Avoidance  |  Press Q to quit")
    print("=" * 60)
    print(f"  Model      : {CFG.MODEL_PATH}")
    print(f"  Device     : {CFG.DEVICE}")
    print(f"  Tracking   : {CFG.ENABLE_TRACKING}")
    print(f"  Conf       : {CFG.CONF_THRESHOLD}")
    print(f"  Danger thr : {CFG.DANGER_THRESHOLD}")
    print(f"  Hysteresis : {CFG.HYSTERESIS_FRAMES} frames")
    print("=" * 60 + "\n")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)

            detections, left_d, right_d, inf_ms = detector.detect(frame)
            decision = engine.decide(robot, left_d, right_d)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            result = FrameResult(
                detections    = detections,
                left_danger   = left_d,
                right_danger  = right_d,
                left_blocked  = left_d  > CFG.DANGER_THRESHOLD,
                right_blocked = right_d > CFG.DANGER_THRESHOLD,
                decision      = decision,
                fps           = fps,
                inference_ms  = inf_ms,
            )

            if not args.no_display:
                vis = draw_frame(frame, result, robot)
                cv2.imshow("YOLO Obstacle Avoidance  |  Q = quit", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('+') or key == ord('='):
                    CFG.DANGER_THRESHOLD = min(CFG.DANGER_THRESHOLD + 0.01, 1.0)
                    print(f"[CONFIG] danger threshold -> {CFG.DANGER_THRESHOLD:.2f}")
                elif key == ord('-'):
                    CFG.DANGER_THRESHOLD = max(CFG.DANGER_THRESHOLD - 0.01, 0.01)
                    print(f"[CONFIG] danger threshold -> {CFG.DANGER_THRESHOLD:.2f}")
            else:
                if int(now * 2) % 2 == 0:
                    objs = [f"{d.label}(#{d.track_id}, {d.distance_m and round(d.distance_m,1)}m)"
                            for d in detections]
                    print(f"[{now:.1f}s] L={left_d:.3f}  R={right_d:.3f}  "
                          f"OBJ={objs}  -> {decision}  ({inf_ms:.0f}ms)")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        robot.stop()
        robot.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
