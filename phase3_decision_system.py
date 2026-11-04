"""
Phase 3 — Harder Challenge Video Only

Model:
    YOLO11 Nano

Video:
    ~/Downloads/harder_challenge_video.mp4

Display:
    LEFT      Video with the accepted lane area and object boxes
    RIGHT     Phase 1 mask: white = marking pixels, gray = line
              candidates, blue / green = chosen boundaries and the
              points supporting them (press M to hide or show)
    BELOW     Status, plus "Lane check": the reason Phase 1 accepted
              or hid the lane in this frame

Controls:
    Space     Pause / resume
    N         Next frame while paused
    M         Show / hide the mask view
    Q or Esc  Quit

Requires the updated Phase 1 (revision 4) and Phase 2 Python files
in the same folder as this script. The header shows which Phase 1
revision was imported.

Distances are approximate.
Lane-change availability remains UNKNOWN.
"""

from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import phase1_lane_detection as phase1
from phase1_lane_detection import LaneDetector, SUPPORT_BINS
from phase2_object_detection import (
    DISPLAY_WIDTH,
    SMOOTHING_WINDOW,
    detect_objects,
    summarise_zones,
)


VIDEO_PATH = (
    Path.home() / "Downloads" / "harder_challenge_video.mp4"
)

# Use the model beside this script if already downloaded.
MODEL_FILE = Path(__file__).resolve().parent / "yolo11n.pt"
MODEL_PATH = str(MODEL_FILE) if MODEL_FILE.is_file() else "yolo11n.pt"

# In this video the car's hood starts at about 93% of the frame
# height, so the lane search stops above it.
LANE_SETTINGS = {
    "roi_bottom": 0.90,
}

PHASE1_VERSION = getattr(phase1, "VERSION", "OLD FILE - replace it")

WINDOW_NAME = "YOLO11 - Harder Challenge"
QUIT_KEYS = (ord("q"), ord("Q"), 27)
SHOW_MASK_AT_START = True

WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
YELLOW = (0, 220, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
GREEN = (0, 220, 0)
BLUE = (255, 0, 0)
PANEL_COLOR = (18, 18, 18)

HEADER_HEIGHT = 32
PANEL_HEIGHT = 215
FRONT_DISTANCE_THRESHOLD = 15.0

# YOLO sometimes labels the car's own hood (plus the road in front of
# it) as a "car" about 1.5 m away. A box this wide that reaches the
# hood is the camera car itself, not traffic.
EGO_BOX_MIN_WIDTH = 0.75     # fraction of frame width
EGO_BOX_MIN_BOTTOM = 0.92    # fraction of frame height


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def resize_frame(frame):
    """Resize while preserving the video's aspect ratio."""
    h, w = frame.shape[:2]
    target_height = max(1, round(h * DISPLAY_WIDTH / w))

    return cv2.resize(frame, (DISPLAY_WIDTH, target_height))


def put_text(frame, text, position, color=WHITE, scale=0.5):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def window_is_open():
    try:
        return cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE
        ) >= 1
    except cv2.error:
        return False


def format_distance(distance):
    if distance is None:
        return "NO DETECTION"

    return f"~{distance:.1f} m"


def distance_color(distance):
    """Color indicates estimated distance, not established safety."""
    if distance is None:
        return GRAY
    if distance <= 15:
        return RED
    if distance <= 30:
        return ORANGE
    return GREEN


def drop_ego_vehicle(detections, frame_shape):
    """Remove detections of the camera car's own hood."""
    h, w = frame_shape[:2]
    kept = []

    for detection in detections:
        x1, _, x2, y2 = detection["bbox"]

        if (
            (x2 - x1) >= EGO_BOX_MIN_WIDTH * w
            and y2 >= EGO_BOX_MIN_BOTTOM * h
        ):
            continue

        kept.append(detection)

    return kept


def boundary_status(lane_info, side):
    if not lane_info[f"{side}_detected"]:
        return "NOT DETECTED"

    if lane_info.get(f"{side}_held"):
        return "HELD"

    return "DETECTED"


def make_assessment(zones, lane_info):
    front = zones["front"]

    if front is not None and front <= FRONT_DISTANCE_THRESHOLD:
        return (
            "FRONT OBJECT NEAR",
            f"Estimated front distance: ~{front:.1f} m",
            ORANGE,
        )

    if not lane_info["lane_area_valid"]:
        return (
            "BOUNDARY DETECTION INCOMPLETE",
            "No accepted lane in this frame (see Lane check)",
            YELLOW,
        )

    if front is None:
        return (
            "NO FRONT OBJECT DETECTED",
            "No tracked object in the fixed front image zone",
            GRAY,
        )

    return (
        "FRONT OBJECT DETECTED",
        f"Estimated front distance: ~{front:.1f} m",
        YELLOW,
    )


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------

def draw_scene(lane_view, detections):
    """Draw boxes on Phase 1's view (boundaries and lane area)."""
    output = lane_view.copy()
    h, w = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        distance = detection["distance_m"]
        confidence = detection["confidence"]
        track_id = detection["track_id"]

        color = distance_color(distance)

        cv2.rectangle(
            output, (x1, y1), (x2, y2), color, 2
        )

        id_text = (
            f" #{track_id}" if track_id is not None else ""
        )

        distance_text = (
            f" ~{distance:.1f}m" if distance is not None else ""
        )

        tag = (
            f"{detection['label']}{id_text} "
            f"{confidence:.0%}{distance_text}"
        )

        scale = 0.45
        (text_w, text_h), baseline = cv2.getTextSize(
            tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )

        text_x = max(0, min(x1, w - text_w - 6))
        text_y = min(
            h - baseline - 3,
            max(text_h + 5, y1 - 6),
        )

        cv2.rectangle(
            output,
            (text_x, text_y - text_h - 4),
            (
                min(w - 1, text_x + text_w + 5),
                text_y + baseline + 3,
            ),
            (20, 20, 20),
            -1,
        )

        put_text(
            output,
            tag,
            (text_x + 2, text_y),
            color,
            scale,
        )

    return output


def add_header(view, video_w, show_mask):
    output = cv2.copyMakeBorder(
        view, HEADER_HEIGHT, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=PANEL_COLOR,
    )

    put_text(output, "VIDEO + LANE + OBJECTS", (12, 21), WHITE, 0.45)

    version_text = f"Phase 1: {PHASE1_VERSION}"
    version_color = GRAY if hasattr(phase1, "VERSION") else RED

    if show_mask:
        put_text(
            output,
            "PHASE 1 MASK | gray: candidates | blue/green: chosen",
            (video_w + 12, 21),
            WHITE,
            0.45,
        )
        (text_w, _), _ = cv2.getTextSize(
            version_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        put_text(
            output, version_text,
            (output.shape[1] - text_w - 12, 21),
            version_color, 0.4,
        )
    else:
        put_text(
            output, version_text, (video_w - 190, 21),
            version_color, 0.4,
        )

    return output


def add_information_panel(
    view,
    video_w,
    zones,
    lane_info,
    detections,
    frame_index,
    paused,
):
    """Place all status information below the video."""
    top_h = view.shape[0]

    output = cv2.copyMakeBorder(
        view, 0, PANEL_HEIGHT, 0, 0,
        cv2.BORDER_CONSTANT, value=PANEL_COLOR,
    )

    assessment, reason, color = make_assessment(zones, lane_info)

    put_text(output, assessment, (15, top_h + 25), color, 0.6)
    put_text(output, reason, (15, top_h + 49), WHITE, 0.46)

    area = "SHOWN" if lane_info["lane_area_valid"] else "HIDDEN"

    put_text(
        output,
        f"Boundaries: LEFT {boundary_status(lane_info, 'left')} | "
        f"RIGHT {boundary_status(lane_info, 'right')} | "
        f"Lane area {area} | Objects: {len(detections)}",
        (15, top_h + 73),
        WHITE,
        0.44,
    )

    # Why the lane is shown or hidden in this frame.
    put_text(
        output,
        f"Lane check: {lane_info['reason']}",
        (15, top_h + 97),
        GREEN if lane_info["lane_area_valid"] else YELLOW,
        0.46,
    )

    fill = lane_info.get("mask_fill")
    fill_text = f" | Mask fill {fill:.0%}" if fill is not None else ""

    put_text(
        output,
        f"Support bins: LEFT {lane_info['left_support']}/{SUPPORT_BINS} | "
        f"RIGHT {lane_info['right_support']}/{SUPPORT_BINS} | "
        f"Candidates: LEFT {lane_info.get('left_candidates', 0)} | "
        f"RIGHT {lane_info.get('right_candidates', 0)}{fill_text}",
        (15, top_h + 121),
        GRAY,
        0.42,
    )

    for index, zone in enumerate(("left", "front", "right")):
        x = 15 + int(index * video_w / 3)

        put_text(
            output,
            f"{zone.upper()}: {format_distance(zones[zone])}",
            (x, top_h + 145),
            WHITE,
            0.43,
        )

    put_text(
        output,
        "Lane changes: UNKNOWN | Approx. distances | Fixed image zones",
        (15, top_h + 169),
        YELLOW,
        0.43,
    )

    state = "PAUSED" if paused else "PLAYING"

    put_text(
        output,
        f"Frame: {frame_index} | {state} | Space: pause/resume | "
        "N: step | M: mask | Q / Esc: quit",
        (15, top_h + 198),
        YELLOW if paused else GRAY,
        0.44,
    )

    return output


def render(result, frame_index, paused, show_mask):
    scene = result["scene"]
    video_w = scene.shape[1]

    view = (
        np.hstack((scene, result["debug"])) if show_mask else scene
    )

    view = add_header(view, video_w, show_mask)

    return add_information_panel(
        view,
        video_w,
        result["zones"],
        result["lane_info"],
        result["detections"],
        frame_index,
        paused,
    )


def size_window(display):
    h, w = display.shape[:2]
    cv2.resizeWindow(WINDOW_NAME, w, h)


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------

def process_frame(
    frame, frame_index, lane_detector, model, histories, last_seen
):
    # Both detectors receive the unannotated frame. Phase 1's
    # returned view (accepted lane only) is the drawing base.
    lane_view, _, _, lane_info = lane_detector.detect(frame)

    # Use Phase 2's detection data, not its HUD image.
    _, detections = detect_objects(
        frame,
        model,
        histories,
        last_seen,
        frame_index,
    )

    detections = drop_ego_vehicle(detections, frame.shape)

    return {
        "scene": draw_scene(lane_view, detections),
        "debug": lane_detector.last_debug.copy(),
        "lane_info": lane_info,
        "zones": summarise_zones(detections),
        "detections": detections,
    }


def run_full_pipeline():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            "Video not found. Place harder_challenge_video.mp4 "
            f"in Downloads.\nExpected path: {VIDEO_PATH}"
        )

    print(f"Phase 1 file: {phase1.__file__}", flush=True)
    print(f"Phase 1 version: {PHASE1_VERSION}", flush=True)
    print(f"Opening video: {VIDEO_PATH}", flush=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV could not open this video: {VIDEO_PATH}"
        )

    histories = defaultdict(
        lambda: deque(maxlen=SMOOTHING_WINDOW)
    )

    lane_detector = LaneDetector(**LANE_SETTINGS)
    last_seen = {}
    frame_index = 0
    exit_reason = "Playback stopped."

    try:
        ret, first_frame = cap.read()

        if not ret:
            raise RuntimeError(
                "Could not decode the first video frame."
            )

        frame = resize_frame(first_frame)

        # Display the selected video before model loading.
        preview = cv2.copyMakeBorder(
            frame, 0, PANEL_HEIGHT, 0, 0,
            cv2.BORDER_CONSTANT, value=PANEL_COLOR,
        )

        put_text(
            preview, "harder_challenge_video.mp4",
            (15, frame.shape[0] + 30), WHITE, 0.55,
        )
        put_text(
            preview, "Loading YOLO11 Nano...",
            (15, frame.shape[0] + 60), YELLOW, 0.6,
        )
        put_text(
            preview, "First inference may take additional time.",
            (15, frame.shape[0] + 90), GRAY, 0.45,
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        size_window(preview)
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(100) & 0xFF

        if key in QUIT_KEYS or not window_is_open():
            print("Stopped before model loading.")
            return

        print(f"Loading model: {MODEL_PATH}", flush=True)
        model = YOLO(MODEL_PATH)

        print("Click the video window for keyboard controls.")
        print("Space: pause/resume | N: step | M: mask | Q / Esc: quit")

        paused = False
        show_mask = SHOW_MASK_AT_START
        result = None
        sized_for = None
        advance = True

        while True:
            if advance:
                if result is not None:
                    ret, next_frame = cap.read()

                    if not ret:
                        exit_reason = (
                            "End of video or no more readable frames."
                        )
                        break

                    frame = resize_frame(next_frame)
                    frame_index += 1

                result = process_frame(
                    frame, frame_index, lane_detector,
                    model, histories, last_seen,
                )

                if frame_index % 30 == 0:
                    zones = result["zones"]
                    print(
                        f"[{frame_index:05d}] "
                        f"Front: {format_distance(zones['front'])} | "
                        f"Left: {format_distance(zones['left'])} | "
                        f"Right: {format_distance(zones['right'])} | "
                        f"Objects: {len(result['detections'])} | "
                        f"Lane: {result['lane_info']['reason']}",
                        flush=True,
                    )

            display = render(result, frame_index, paused, show_mask)

            if sized_for != show_mask:
                size_window(display)
                sized_for = show_mask

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30 if paused else 1) & 0xFF

            if key in QUIT_KEYS:
                exit_reason = "Stopped by keyboard."
                break

            if not window_is_open():
                exit_reason = "Video window closed."
                break

            if key == ord(" "):
                paused = not paused
                print("Paused." if paused else "Resumed.")

            if key in (ord("m"), ord("M")):
                show_mask = not show_mask

            advance = not paused

            if paused and key in (ord("n"), ord("N")):
                advance = True

        print(exit_reason)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    run_full_pipeline()