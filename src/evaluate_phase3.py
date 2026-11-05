"""
Phase 3 evaluation: fixed image zones vs lane-based zones.

For every frame this runs Phase 1 (lanes) and Phase 2 (YOLO + tracking)
the same way Phase 3 does, removes the camera car's hood box, and then
decides which objects are "in front" in two ways:

  fixed  Phase 2's middle slice of the image (35% to 65% of the width),
         which is what Phase 3 uses today
  lane   the object's bottom-centre point (where it meets the road) lies
         between the accepted Phase 1 lane boundaries; when no lane is
         accepted, or the object sits beyond the lane area, it falls back
         to the fixed rule

It also checks whether tracking hides objects: Phase 2 only reports boxes
that ByteTrack has confirmed as tracks, so the same frame is also run
through plain YOLO detection (a second model instance) and any detected
object with no matching tracked box is counted as missed by tracking.

It prints how often each rule raises FRONT OBJECT NEAR, how often they
disagree, how many objects tracking hid, timing, and saves contact sheets
of the frames where the zone rules disagree and where tracking missed.

Usage:
    python src/evaluate_phase3.py videos/lane_video.mp4
    python src/evaluate_phase3.py videos/harder_challenge_video.mp4 --roi-bottom 0.90
"""

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import phase1_lane_detection as p1
import phase2_object_detection as p2
from phase3_decision_system import FRONT_DISTANCE_THRESHOLD, drop_ego_vehicle

_MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "yolo11n.pt"
DEFAULT_MODEL = str(_MODEL_FILE) if _MODEL_FILE.is_file() else "yolo11n.pt"

SHEET_TILES = 12
TILE_WIDTH = 480


def lane_zone(detection, left_pts, right_pts):
    """'left', 'front' or 'right' from the accepted lane, or None if unknown."""
    if left_pts is None or right_pts is None:
        return None

    x1, _, x2, y2 = detection["bbox"]
    foot_x = (x1 + x2) / 2.0

    top = max(left_pts[0, 1], right_pts[0, 1])
    bottom = min(left_pts[-1, 1], right_pts[-1, 1])

    if y2 < top:
        return None              # base is beyond the lane area: can't tell

    y = min(y2, bottom)          # base below the area (very close): use its edge
    left_x = np.interp(y, left_pts[:, 1], left_pts[:, 0])
    right_x = np.interp(y, right_pts[:, 1], right_pts[:, 0])

    if foot_x < left_x:
        return "left"
    if foot_x > right_x:
        return "right"
    return "front"


def raw_detections(frame, model):
    """Plain YOLO detections (no tracking), built the same way as Phase 2."""
    h, w = frame.shape[:2]
    result = model.predict(frame, classes=list(p2.TRACKED_CLASSES),
                           conf=p2.CONFIDENCE_THRESHOLD, verbose=False)[0]
    found = []
    if result.boxes is None or len(result.boxes) == 0:
        return found
    coords = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), cls in zip(coords, classes):
        label = p2.TRACKED_CLASSES.get(int(cls))
        x1, x2 = int(max(0, min(w - 1, x1))), int(max(0, min(w - 1, x2)))
        y1, y2 = int(max(0, min(h - 1, y1))), int(max(0, min(h - 1, y2)))
        if label is None or x2 <= x1 or y2 <= y1:
            continue
        found.append({"label": label, "bbox": (x1, y1, x2, y2),
                      "distance_m": p2.estimate_distance(label, x2 - x1),
                      "zone": p2.classify_zone((x1 + x2) / 2, w)})
    return drop_ego_vehicle(found, frame.shape)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw_miss_tile(view, tracked, missed, index):
    tile = view.copy()
    h, w = tile.shape[:2]
    for d, color in [(d, (0, 200, 0)) for d in tracked] + [(d, (0, 220, 255)) for d in missed]:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 3)
    for d in missed:
        x1, y1 = map(int, d["bbox"][:2])
        cv2.putText(tile, f"{d['label']} ~{d['distance_m']:.0f}m", (x1, max(48, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.putText(tile, f"frame {index}   yellow: detected but not tracked   green: tracked",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(tile, (TILE_WIDTH, round(h * TILE_WIDTH / w)))


def front_near(detections, key):
    distances = [d["distance_m"] for d in detections if d[key] == "front"]
    return bool(distances) and min(distances) <= FRONT_DISTANCE_THRESHOLD


def draw_tile(view, detections, index, fixed_alert, lane_alert):
    tile = view.copy()
    h, w = tile.shape[:2]

    for ratio in (p2.LEFT_ZONE_RATIO, p2.RIGHT_ZONE_RATIO):
        x = int(w * ratio)
        cv2.line(tile, (x, 0), (x, h - 1), (160, 160, 160), 1)

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = (0, 0, 255) if d["zone_lane"] == "front" else (200, 200, 200)
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)
        cv2.circle(tile, ((x1 + x2) // 2, y2), 5, color, -1)

    label = (f"frame {index}   fixed: {'NEAR' if fixed_alert else '-'}   "
             f"lane: {'NEAR' if lane_alert else '-'}")
    cv2.rectangle(tile, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    scale = TILE_WIDTH / w
    return cv2.resize(tile, (TILE_WIDTH, round(h * scale)))


def contact_sheet(tiles, path):
    if not tiles:
        return False
    picks = np.linspace(0, len(tiles) - 1, min(SHEET_TILES, len(tiles))).round().astype(int)
    chosen = [tiles[i] for i in picks]
    while len(chosen) % 3:
        chosen.append(np.zeros_like(chosen[0]))
    rows = [np.hstack(chosen[i:i + 3]) for i in range(0, len(chosen), 3)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--roi-bottom", type=float, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    video = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")

    detector = p1.LaneDetector(**({"roi_bottom": args.roi_bottom} if args.roi_bottom else {}))
    model = YOLO(args.model)
    detector_model = YOLO(args.model)    # separate instance: plain detection
    histories = defaultdict(lambda: deque(maxlen=p2.SMOOTHING_WINDOW))
    last_seen = {}

    frames = lane_frames = objects = lane_decided = 0
    fixed_alerts = lane_alerts = only_fixed = only_lane = 0
    front_to_side = side_to_front = 0
    lane_ms, yolo_ms, total_ms = [], [], []
    tiles = []
    miss_tiles = []
    raw_objects = missed_objects = miss_frames = missed_near_front = 0

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (p2.DISPLAY_WIDTH, max(1, round(h * p2.DISPLAY_WIDTH / w))))

        t0 = time.perf_counter()
        view, left_pts, right_pts, info = detector.detect(frame)
        lane_ms.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        _, detections = p2.detect_objects(frame, model, histories, last_seen, index)
        yolo_ms.append((time.perf_counter() - t1) * 1000)
        detections = drop_ego_vehicle(detections, frame.shape)

        valid = info["lane_area_valid"]
        lane_frames += valid
        for d in detections:
            d["zone_fixed"] = d["zone"]
            lz = lane_zone(d, left_pts, right_pts) if valid else None
            d["zone_lane"] = lz if lz is not None else d["zone"]
            lane_decided += lz is not None
            if d["zone_fixed"] == "front" and d["zone_lane"] != "front":
                front_to_side += 1
            if d["zone_fixed"] != "front" and d["zone_lane"] == "front":
                side_to_front += 1
        objects += len(detections)

        raw = raw_detections(frame, detector_model)
        raw_objects += len(raw)
        missed = [r for r in raw
                  if all(iou(r["bbox"], d["bbox"]) < 0.3 for d in detections)]
        if missed:
            missed_objects += len(missed)
            miss_frames += 1
            for r in missed:
                lz = lane_zone(r, left_pts, right_pts) if valid else None
                if (lz or r["zone"]) == "front" and r["distance_m"] <= FRONT_DISTANCE_THRESHOLD:
                    missed_near_front += 1
            if len(miss_tiles) < 400:
                miss_tiles.append(draw_miss_tile(view, detections, missed, index))

        fixed_alert = front_near(detections, "zone_fixed")
        lane_alert = front_near(detections, "zone_lane")
        fixed_alerts += fixed_alert
        lane_alerts += lane_alert
        only_fixed += fixed_alert and not lane_alert
        only_lane += lane_alert and not fixed_alert

        if fixed_alert != lane_alert and len(tiles) < 400:
            tiles.append(draw_tile(view, detections, index, fixed_alert, lane_alert))

        total_ms.append((time.perf_counter() - start) * 1000)
        frames += 1
        index += 1

    cap.release()
    if not frames:
        raise SystemExit("No frames read.")

    sheet = Path(f"phase3_disagreements_{video.stem}.jpg")
    saved = contact_sheet(tiles, sheet)

    print(f"\nVideo: {video.name}  ({frames} frames)")
    print(f"Lane area accepted: {lane_frames} frames ({lane_frames / frames:.1%})")
    print(f"Objects after hood filter: {objects} ({objects / frames:.2f} per frame)")
    if objects:
        print(f"Objects the lane rule could place: {lane_decided} ({lane_decided / objects:.1%}); "
              "the rest used the fixed rule")
    print(f"  fixed 'front' -> lane says left/right: {front_to_side}")
    print(f"  fixed left/right -> lane says 'front': {side_to_front}")
    print(f"\nFRONT OBJECT NEAR (estimated <= {FRONT_DISTANCE_THRESHOLD:.0f} m):")
    print(f"  fixed zones: {fixed_alerts} frames ({fixed_alerts / frames:.1%})")
    print(f"  lane zones:  {lane_alerts} frames ({lane_alerts / frames:.1%})")
    print(f"  only fixed alerted: {only_fixed}   only lane alerted: {only_lane}")
    if saved:
        print(f"Saved {min(SHEET_TILES, len(tiles))} of {len(tiles)} disagreement frames to {sheet}")
    else:
        print("No disagreement frames.")

    print(f"\nTracking check (plain YOLO detection vs Phase 2's tracked boxes):")
    print(f"  objects detected: {raw_objects}   tracked: {objects}")
    print(f"  detected but not tracked: {missed_objects} objects in {miss_frames} frames "
          f"({miss_frames / frames:.1%} of frames)")
    print(f"  of those, in front and within {FRONT_DISTANCE_THRESHOLD:.0f} m: {missed_near_front}")
    miss_sheet = Path(f"phase3_tracking_misses_{video.stem}.jpg")
    if contact_sheet(miss_tiles, miss_sheet):
        print(f"Saved {min(SHEET_TILES, len(miss_tiles))} of {len(miss_tiles)} tracking-miss frames to {miss_sheet}")

    def summary(name, values):
        v = np.array(values)
        print(f"  {name:<10} median {np.median(v):6.1f} ms   p95 {np.percentile(v, 95):6.1f} ms")

    print("\nTiming (this machine, 960-pixel frames):")
    summary("lane", lane_ms)
    summary("YOLO+track", yolo_ms)
    summary("full frame", total_ms)
    print(f"  (full frame includes the extra detection pass used only by this check)")


if __name__ == "__main__":
    main()
