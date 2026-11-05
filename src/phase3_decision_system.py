"""
Phase 3 - Lanes + objects + decision

Usage:
    python src/phase3_decision_system.py                      # harder_challenge_video.mp4
    python src/phase3_decision_system.py videos/lane_video.mp4 --roi-bottom 0.93
    python src/phase3_decision_system.py VIDEO --fixed-zones  # old image-slice zones
    python src/phase3_decision_system.py VIDEO --no-smoothing # raw Phase 1 lines

What it adds on top of Phase 1 (lanes) and Phase 2 (YOLO + tracking):
    * Smooth lane drawing: each accepted boundary is sampled at fixed
      image rows and blended with the previous frame, so the lines do
      not jump around. When the lane keeps moving the same way (a real
      curve or lane change) the blend follows faster. If Phase 1 hides
      the lane for 1-3 frames, the last lane stays on screen, drawn
      fainter and labelled as held; after that nothing is drawn.
    * Lane-based zones: an object is "front" when the point where it
      meets the road lies inside the accepted lane, "left"/"right" when
      it lies outside. Without an accepted lane (or for objects beyond
      the lane area) the old fixed image slices are used.
    * One status per frame, in priority order: FRONT OBJECT NEAR,
      BOUNDARY DETECTION INCOMPLETE, FRONT OBJECT DETECTED / NONE.

Display:
    LEFT      Video with the smoothed lane area and object boxes
    RIGHT     Phase 1 mask (press M to hide or show)
    BELOW     Status, lane check and zone method

Controls:
    Space     Pause / resume
    N         Next frame while paused
    M         Show / hide the mask view
    Q or Esc  Quit

Requires Phase 1 (revision 5) and Phase 2 in the same folder.
Distances are approximate. Lane-change availability remains UNKNOWN.
"""

import argparse
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import phase1_lane_detection as phase1
from phase1_lane_detection import (
    LaneDetector,
    LaneSmoother,
    SUPPORT_BINS,
    draw_smoothed_lane,
)
from phase2_object_detection import (
    DISPLAY_WIDTH,
    SMOOTHING_WINDOW,
    detect_objects,
    summarise_zones,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LOCAL_VIDEO = PROJECT_DIR / "videos" / "harder_challenge_video.mp4"
DEFAULT_VIDEO = (
    LOCAL_VIDEO if LOCAL_VIDEO.is_file()
    else Path.home() / "Downloads" / "harder_challenge_video.mp4"
)

# Use the model beside this script if already downloaded.
MODEL_FILE = PROJECT_DIR / "models" / "yolo11n.pt"
MODEL_PATH = str(MODEL_FILE) if MODEL_FILE.is_file() else "yolo11n.pt"

# In this video the car's hood starts at about 93% of the frame
# height, so the lane search stops above it.
DEFAULT_ROI_BOTTOM = 0.90

PHASE1_VERSION = getattr(phase1, "VERSION", "OLD FILE - replace it")

WINDOW_NAME = "Phase 3 - Lanes + Objects"
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


def lane_zone(detection, left_pts, right_pts):
    """'left', 'front' or 'right' from the lane, or None if unknown."""
    if left_pts is None or right_pts is None:
        return None

    x1, _, x2, y2 = detection["bbox"]
    foot_x = (x1 + x2) / 2.0

    top = max(left_pts[0, 1], right_pts[0, 1])
    bottom = min(left_pts[-1, 1], right_pts[-1, 1])

    if y2 < top:
        return None              # beyond the lane area: can't tell

    y = min(y2, bottom)          # very close: use the lane's bottom edge
    left_x = np.interp(y, left_pts[:, 1], left_pts[:, 0])
    right_x = np.interp(y, right_pts[:, 1], right_pts[:, 0])

    if foot_x < left_x:
        return "left"
    if foot_x > right_x:
        return "right"
    return "front"


def assign_zones(detections, left_pts, right_pts, use_lane):
    """Re-zone objects by the lane where possible; return a panel note."""
    if not use_lane:
        return "Fixed image zones"

    placed = 0
    for detection in detections:
        zone = lane_zone(detection, left_pts, right_pts)
        if zone is not None:
            detection["zone"] = zone
            placed += 1

    if not detections:
        return "Zones: by lane when shown"
    return (f"Zones: by lane {placed}/{len(detections)}, "
            f"image slice {len(detections) - placed}/{len(detections)}")


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
            "No tracked object in the front zone",
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
    zone_note="Fixed image zones",
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
        f"Lane changes: UNKNOWN | Approx. distances | {zone_note}",
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
        result.get("zone_note", "Fixed image zones"),
    )


def size_window(display):
    h, w = display.shape[:2]
    cv2.resizeWindow(WINDOW_NAME, w, h)


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------

def process_frame(
    frame, frame_index, lane_detector, model, histories, last_seen,
    smoother, options,
):
    # Both detectors receive the unannotated frame.
    lane_view, left_pts, right_pts, lane_info = lane_detector.detect(frame)

    # Use Phase 2's detection data, not its HUD image.
    _, detections = detect_objects(
        frame,
        model,
        histories,
        last_seen,
        frame_index,
    )

    detections = drop_ego_vehicle(detections, frame.shape)
    h, w = frame.shape[:2]
    valid = lane_info["lane_area_valid"]

    held = False
    if options["smooth"]:
        left_s, right_s, held = smoother.update(
            left_pts, right_pts, valid,
            lane_detector.top * h, lane_detector.bottom * h, w,
        )
        if left_s is not None and right_s is not None:
            base = draw_smoothed_lane(frame, left_s, right_s, held)
        else:
            left_s = right_s = None
            base = frame.copy()
    else:
        left_s, right_s = (left_pts, right_pts) if valid else (None, None)
        base = lane_view

    if held:
        lane_info = dict(
            lane_info,
            reason=lane_info["reason"] + " | showing last lane (held)",
        )

    # Zones only trust a lane Phase 1 accepted in this frame.
    zone_note = assign_zones(
        detections,
        None if held else left_s,
        None if held else right_s,
        options["lane_zones"],
    )

    return {
        "scene": draw_scene(base, detections),
        "debug": lane_detector.last_debug.copy(),
        "lane_info": lane_info,
        "zones": summarise_zones(detections),
        "detections": detections,
        "zone_note": zone_note,
    }


def run_full_pipeline(
    video_path=DEFAULT_VIDEO,
    model_path=MODEL_PATH,
    roi_bottom=DEFAULT_ROI_BOTTOM,
    lane_zones=True,
    smooth=True,
):
    video_path = Path(video_path).expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Phase 1 file: {phase1.__file__}", flush=True)
    print(f"Phase 1 version: {PHASE1_VERSION}", flush=True)
    print(f"Opening video: {video_path}", flush=True)
    print(f"Zones: {'lane-based' if lane_zones else 'fixed image slices'} | "
          f"Lane smoothing: {'on' if smooth else 'off'} | "
          f"ROI bottom: {roi_bottom}", flush=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV could not open this video: {video_path}"
        )

    histories = defaultdict(
        lambda: deque(maxlen=SMOOTHING_WINDOW)
    )

    lane_detector = LaneDetector(roi_bottom=roi_bottom)
    smoother = LaneSmoother()
    options = {"lane_zones": lane_zones, "smooth": smooth}
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
            preview, video_path.name,
            (15, frame.shape[0] + 30), WHITE, 0.55,
        )
        put_text(
            preview, f"Loading {Path(model_path).name}...",
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

        print(f"Loading model: {model_path}", flush=True)
        model = YOLO(model_path)

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
                    smoother, options,
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
    parser = argparse.ArgumentParser(description="Phase 3: lanes + objects.")
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO),
                        help="Video path (default: harder_challenge_video.mp4)")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO weights")
    parser.add_argument("--roi-bottom", type=float, default=DEFAULT_ROI_BOTTOM,
                        help="Lane search bottom, fraction of frame height "
                             "(0.90 stays above a visible hood)")
    parser.add_argument("--fixed-zones", action="store_true",
                        help="Use the old fixed image slices for zones")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Draw Phase 1's raw lines without smoothing")
    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video,
        model_path=args.model,
        roi_bottom=args.roi_bottom,
        lane_zones=not args.fixed_zones,
        smooth=not args.no_smoothing,
    )