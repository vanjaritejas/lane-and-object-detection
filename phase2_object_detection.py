"""
Phase 2 — Object Detection & Approximate Distance Estimation

Features:
- YOLO object detection and tracking.
- Distance smoothing for objects with valid track IDs.
- Fixed left, front, and right image zones.
- Bounding boxes, confidence scores, and distance estimates.
- HUD showing the nearest estimated object per zone.

Limitations:
- Image zones are not detected road lanes.
- Distances use assumed widths and an uncalibrated focal length.
- No detection does not establish that a zone is clear.
"""

import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DISPLAY_WIDTH = 960
CONFIDENCE_THRESHOLD = 0.35

# Approximate object widths in metres.
KNOWN_WIDTHS_M = {
    "person": 0.5,
    "car": 1.8,
    "motorcycle": 0.8,
    "bus": 2.5,
    "truck": 2.5,
    "cat": 0.25,
    "dog": 0.4,
    "horse": 0.8,
    "sheep": 0.5,
    "cow": 0.9,
}

# Placeholder focal length for frames resized to DISPLAY_WIDTH.
# Camera calibration is needed for meaningful metric distances.
FOCAL_LENGTH_PX = 800.0

NEAR_DISTANCE_M = 15.0
MID_DISTANCE_M = 30.0

# COCO class IDs used by yolov8n.pt.
TRACKED_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
}

LEFT_ZONE_RATIO = 0.35
RIGHT_ZONE_RATIO = 0.65

SMOOTHING_WINDOW = 8
HISTORY_MAX_AGE = 60

GREEN = (0, 220, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
GRAY = (180, 180, 180)
WHITE = (255, 255, 255)


# --------------------------------------------------
# Distance estimation
# --------------------------------------------------

def estimate_distance(label, bbox_width_px):
    """Estimate distance using assumed width and focal length."""
    if bbox_width_px <= 0:
        return None

    known_width = KNOWN_WIDTHS_M[label]
    return float(known_width * FOCAL_LENGTH_PX / bbox_width_px)


def smooth_distance(track_id, raw_distance, histories):
    """Smooth only objects that have a valid tracker ID."""
    if track_id is None:
        return raw_distance

    histories[track_id].append(raw_distance)
    return float(np.mean(histories[track_id]))


def distance_color(distance):
    """Color by estimated distance; green does not establish safety."""
    if distance is None:
        return GRAY
    if distance <= NEAR_DISTANCE_M:
        return RED
    if distance <= MID_DISTANCE_M:
        return ORANGE
    return GREEN


# --------------------------------------------------
# Fixed image zones
# --------------------------------------------------

def classify_zone(center_x, frame_width):
    left_boundary = int(frame_width * LEFT_ZONE_RATIO)
    right_boundary = int(frame_width * RIGHT_ZONE_RATIO)

    if center_x < left_boundary:
        return "left"
    if center_x > right_boundary:
        return "right"
    return "front"


def draw_zones(frame):
    h, w = frame.shape[:2]

    left_boundary = int(w * LEFT_ZONE_RATIO)
    right_boundary = int(w * RIGHT_ZONE_RATIO)

    cv2.line(
        frame,
        (left_boundary, 0),
        (left_boundary, h - 1),
        GRAY,
        1,
    )
    cv2.line(
        frame,
        (right_boundary, 0),
        (right_boundary, h - 1),
        GRAY,
        1,
    )

    labels = [
        ("LEFT ZONE", 10),
        ("FRONT ZONE", left_boundary + 10),
        ("RIGHT ZONE", right_boundary + 10),
    ]

    for text, x in labels:
        cv2.putText(
            frame, text, (x, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, WHITE, 1, cv2.LINE_AA,
        )

    return frame


def summarise_zones(detections):
    """Find the nearest estimated object in each image zone."""
    summary = {"left": None, "front": None, "right": None}

    for detection in detections:
        zone = detection["zone"]
        distance = detection["distance_m"]

        if summary[zone] is None or distance < summary[zone]:
            summary[zone] = distance

    return summary


# --------------------------------------------------
# Drawing helpers
# --------------------------------------------------

def draw_detection(frame, detection):
    x1, y1, x2, y2 = detection["bbox"]
    distance = detection["distance_m"]
    confidence = detection["confidence"]
    track_id = detection["track_id"]
    label = detection["label"]
    color = distance_color(distance)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    id_text = f" #{track_id}" if track_id is not None else ""
    tag = f"{label}{id_text} {confidence:.0%} ~{distance:.1f}m"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        tag, font, scale, thickness
    )

    h, w = frame.shape[:2]
    text_x = max(0, min(x1, w - text_width - 8))
    text_y = min(h - baseline - 4, max(y1 - 6, text_height + 6))

    cv2.rectangle(
        frame,
        (text_x, text_y - text_height - 4),
        (min(w - 1, text_x + text_width + 6), text_y + baseline + 3),
        color,
        -1,
    )

    cv2.putText(
        frame, tag, (text_x + 3, text_y),
        font, scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )


def draw_hud(frame, zones, object_count):
    h, w = frame.shape[:2]
    panel_top = max(0, h - 90)

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, panel_top), (w - 1, h - 1),
        (15, 15, 15), -1,
    )
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    for index, zone in enumerate(("left", "front", "right")):
        x = int(index * w / 3) + 15
        distance = zones[zone]

        text = (
            f"~{distance:.1f} m"
            if distance is not None
            else "NO DETECTION"
        )

        cv2.putText(
            frame, zone.upper(), (x, h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, GRAY, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (x, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, distance_color(distance), 2, cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Approx. distances | Fixed image zones | Not a lane-change decision",
        (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, GRAY, 1, cv2.LINE_AA,
    )

    cv2.putText(
        frame, f"Objects: {object_count}", (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, WHITE, 1, cv2.LINE_AA,
    )

    front_distance = zones["front"]
    if front_distance is not None and front_distance <= NEAR_DISTANCE_M:
        cv2.putText(
            frame,
            "FRONT OBJECT: LOW ESTIMATED DISTANCE",
            (15, max(75, panel_top - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, RED, 2, cv2.LINE_AA,
        )

    return frame


# --------------------------------------------------
# Object detection
# --------------------------------------------------

def detect_objects(frame, model, histories, last_seen, frame_index):
    h, w = frame.shape[:2]

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(TRACKED_CLASSES),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]

    detections = []
    annotated = draw_zones(frame.copy())

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        # Transfer all box data to CPU once per frame.
        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else None
        )

        for index, coords in enumerate(coordinates):
            class_id = int(class_ids[index])
            label = TRACKED_CLASSES.get(class_id)

            if label is None:
                continue

            x1, y1, x2, y2 = map(float, coords)

            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            track_id = (
                int(track_ids[index])
                if track_ids is not None
                else None
            )

            raw_distance = estimate_distance(label, x2 - x1)
            if raw_distance is None:
                continue

            distance = smooth_distance(
                track_id, raw_distance, histories
            )

            if track_id is not None:
                last_seen[track_id] = frame_index

            detection = {
                "label": label,
                "distance_m": distance,
                "zone": classify_zone((x1 + x2) / 2, w),
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confidences[index]),
                "track_id": track_id,
            }

            detections.append(detection)
            draw_detection(annotated, detection)

    # Remove histories for tracks absent for a while.
    expired_ids = [
        track_id
        for track_id, seen_frame in last_seen.items()
        if frame_index - seen_frame > HISTORY_MAX_AGE
    ]

    for track_id in expired_ids:
        histories.pop(track_id, None)
        last_seen.pop(track_id, None)

    zones = summarise_zones(detections)
    annotated = draw_hud(annotated, zones, len(detections))

    return annotated, detections


# --------------------------------------------------
# Run on video
# --------------------------------------------------

def run_phase2(video_path, model_path="yolov8n.pt"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    histories = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
    last_seen = {}

    try:
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        print("Phase 2 running — press Q or Esc to quit.")

        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video or no more readable frames.")
                break

            # Preserve the original aspect ratio.
            original_h, original_w = frame.shape[:2]
            display_height = max(
                1, round(original_h * DISPLAY_WIDTH / original_w)
            )
            frame = cv2.resize(
                frame, (DISPLAY_WIDTH, display_height)
            )

            annotated, detections = detect_objects(
                frame, model, histories, last_seen, frame_index
            )

            # Avoid printing a line for every frame.
            if frame_index % 30 == 0:
                zones = summarise_zones(detections)

                def format_distance(value):
                    return (
                        f"~{value:.1f}m"
                        if value is not None
                        else "NO DETECTION"
                    )

                print(
                    f"Front: {format_distance(zones['front'])} | "
                    f"Left: {format_distance(zones['left'])} | "
                    f"Right: {format_distance(zones['right'])} | "
                    f"Objects: {len(detections)}"
                )

            cv2.imshow("Phase 2 - Object Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            frame_index += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object detection with approximate distances."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="lane_video.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to COCO-trained YOLO weights.",
    )

    args = parser.parse_args()
    run_phase2(args.video, args.model)