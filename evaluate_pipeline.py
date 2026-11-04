"""
Headless evaluation for the driver-assistance prototype.

Runs Phase 1 (and optionally Phase 2 YOLO) on a video without opening
windows and prints:
  * how often a lane area was accepted, and why it was hidden otherwise
  * the longest stretch of hidden frames
  * optional: whether the accepted LEFT boundary sits on a yellow line
    (a simple accuracy proxy for roads with a yellow left edge line)
  * per-frame timing for lane detection and YOLO, and overall FPS
  * optional: how often the "FRONT OBJECT NEAR" alert fired

Examples:
    python evaluate_pipeline.py lane_video.mp4
    python evaluate_pipeline.py random.mp4 --yellow-check
    python evaluate_pipeline.py random.mp4 --model yolov8n.pt
    python evaluate_pipeline.py harder.mp4 --roi-bottom 0.90 --model yolo11n.pt

Timing depends on the machine; report the hardware with any FPS figure.
"""

import argparse
import collections
import time

import cv2
import numpy as np

import phase1_lane_detection as p1


def yellow_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([p1.YELLOW_HUE[0], p1.YELLOW_MIN_SATURATION, p1.YELLOW_MIN_VALUE], np.uint8)
    high = np.array([p1.YELLOW_HUE[1], 255, 255], np.uint8)
    mask = cv2.inRange(hsv, low, high)
    return cv2.dilate(mask, np.ones((1, 9), np.uint8))  # about +-4 px tolerance


def left_on_yellow(frame, left_points, detector):
    """Fraction of the lower left boundary that lies on yellow paint, or None."""
    h, w = frame.shape[:2]
    mask = yellow_mask(frame)
    band = np.zeros_like(mask)
    band[int(h * 0.75):int(h * detector.bottom)] = 255
    visible = cv2.bitwise_and(cv2.bitwise_and(mask, detector.roi_mask(h, w)), band)
    if np.count_nonzero(visible) <= 150 or left_points is None:
        return None
    pts = left_points[(left_points[:, 1] >= h * 0.75)
                      & (left_points[:, 0] >= 0) & (left_points[:, 0] < w)]
    if not len(pts):
        return None
    return float(np.mean(mask[pts[:, 1], pts[:, 0]] > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--roi-bottom", type=float, default=None)
    parser.add_argument("--model", default=None, help="YOLO weights; omit to skip Phase 2")
    parser.add_argument("--yellow-check", action="store_true")
    args = parser.parse_args()

    model = None
    if args.model:
        from collections import defaultdict, deque
        from ultralytics import YOLO
        import phase2_object_detection as p2
        model = YOLO(args.model)
        histories = defaultdict(lambda: deque(maxlen=p2.SMOOTHING_WINDOW))
        last_seen = {}
        try:
            from phase3_decision_system import drop_ego_vehicle
        except Exception:
            drop_ego_vehicle = None

    detector = p1.LaneDetector(**({"roi_bottom": args.roi_bottom} if args.roi_bottom else {}))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    frames = accepted = held = 0
    reasons = collections.Counter()
    lane_ms, yolo_ms, total_ms = [], [], []
    hidden_run = longest_hidden = 0
    yellow_good = yellow_bad = 0
    bad_frames = []
    front_alerts = 0
    detections_total = 0

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        original = frame
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (p1.DISPLAY_WIDTH, max(1, round(h * p1.DISPLAY_WIDTH / w))))

        t0 = time.perf_counter()
        _, left_pts, _, info = detector.detect(frame)
        lane_ms.append((time.perf_counter() - t0) * 1000)

        frames += 1
        if info["lane_area_valid"]:
            accepted += 1
            held += bool(info["left_held"] or info["right_held"])
            hidden_run = 0
        else:
            hidden_run += 1
            longest_hidden = max(longest_hidden, hidden_run)
        reasons[info["reason"].split(" (holding")[0]] += 1

        if args.yellow_check and info["lane_area_valid"]:
            frac = left_on_yellow(frame, left_pts, detector)
            if frac is not None:
                if frac >= 0.5:
                    yellow_good += 1
                else:
                    yellow_bad += 1
                    bad_frames.append(index)

        if model is not None:
            # Phase 2 runs on its own display width, as in Phase 3.
            obj_frame = cv2.resize(original, (p2.DISPLAY_WIDTH, max(1, round(h * p2.DISPLAY_WIDTH / w))))
            t1 = time.perf_counter()
            _, detections = p2.detect_objects(obj_frame, model, histories, last_seen, index)
            yolo_ms.append((time.perf_counter() - t1) * 1000)
            if drop_ego_vehicle is not None:
                detections = drop_ego_vehicle(detections, obj_frame.shape)
            detections_total += len(detections)
            front = p2.summarise_zones(detections)["front"]
            if front is not None and front <= p2.NEAR_DISTANCE_M:
                front_alerts += 1

        total_ms.append((time.perf_counter() - start) * 1000)
        index += 1

    cap.release()
    if not frames:
        raise SystemExit("No frames read.")

    print(f"\nVideo: {args.video}  ({frames} frames)")
    print(f"Lane area accepted: {accepted}/{frames} ({accepted / frames:.1%}); "
          f"{held} of those were holding one side from earlier frames")
    print(f"Longest stretch of hidden frames: {longest_hidden}")
    print("Reasons:")
    for reason, count in reasons.most_common():
        print(f"  {count:5d}  {reason}")

    if args.yellow_check:
        checked = yellow_good + yellow_bad
        if checked:
            print(f"\nYellow-line check on {checked} accepted frames with visible yellow paint:")
            print(f"  left boundary on yellow line: {yellow_good} ({yellow_good / checked:.1%})")
            print(f"  left boundary off the line:   {yellow_bad} ({yellow_bad / checked:.1%})")
            print(f"  off-line frames: {bad_frames[:80]}")
            print("  (Spot-check some of these frames by eye before quoting the figure.)")
        else:
            print("\nYellow-line check: no accepted frames with visible yellow paint.")

    def summary(name, values):
        v = np.array(values)
        print(f"  {name:<14} median {np.median(v):6.1f} ms   p95 {np.percentile(v, 95):6.1f} ms")

    print("\nTiming (this machine):")
    summary("lane detection", lane_ms)
    if yolo_ms:
        summary("YOLO + tracking", yolo_ms)
        print(f"  objects per frame (after ego filter): {detections_total / frames:.2f}")
        print(f"  frames with FRONT OBJECT NEAR alert: {front_alerts} ({front_alerts / frames:.1%})")
    summary("full frame", total_ms)
    print(f"  overall throughput: ~{1000 / np.median(total_ms):.0f} FPS")


if __name__ == "__main__":
    main()
