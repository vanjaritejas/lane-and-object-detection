"""
PHASE 1 — LANE BOUNDARIES AND ESTIMATED LANE AREA  (revision 4)

How it works:
    * Marking pixels: yellow + white, with white thresholds that rise
      with road brightness so sunlit or washed-out asphalt does not
      flood the mask. A frame whose mask is still mostly "marking" is
      treated as unreadable instead of being guessed.
    * Candidates: whole perspective lines from Hough voting on marking
      centerlines (dash gaps do not matter), then each line is grown
      upward along the marking so curves are followed.
    * Ego lane: the INNERMOST valid left/right pair. Nearly vertical
      lines in the middle of the lane are rejected.
    * Pairs must share a plausible vanishing point and lane width; the
      width is learned over time.
    * A line must have clean road on its lane side; lines found inside
      gravel, dirt or foliage texture are rejected.
    * A boundary that was part of an accepted lane and goes missing for
      a few frames is held (and reported as held). Lines are drawn on
      the video only when the lane area is accepted.

The region of interest can be set per video:
    LaneDetector(roi_bottom=0.89)          # stop above a visible hood

Default video search:
    1. lane_video.mp4 beside this Python file
    2. ~/Downloads/lane_video.mp4

Controls:
    SPACE     Pause / resume
    N         Next frame while paused
    R         Restart
    Q / ESC   Quit

Display:
    LEFT      Video with estimated boundaries and lane area
    RIGHT     Mask, every line candidate (gray), chosen boundaries and
              the centerline points supporting them

Requires:
    pip install opencv-python numpy

The estimated region does not indicate lane-change safety.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# Shown by Phase 3 so you can confirm which Phase 1 file is imported.
VERSION = "revision 4"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_VIDEO = SCRIPT_DIR / "lane_video.mp4"
DOWNLOAD_VIDEO = Path.home() / "Downloads" / "lane_video.mp4"

DEFAULT_VIDEO = (
    LOCAL_VIDEO if LOCAL_VIDEO.is_file() else DOWNLOAD_VIDEO
)

DISPLAY_WIDTH = 640
PLAYBACK_DELAY_MS = 90

WINDOW_NAME = "Phase 1 - Lane Area Diagnostics"
QUIT_KEYS = (ord("q"), ord("Q"), 27)

BLUE = (255, 100, 0)
GREEN = (0, 230, 0)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DIM_GRAY = (110, 110, 110)
DARK_GRAY = (60, 60, 60)

# Default road region for a forward-facing camera (fractions of the
# frame). Override per video through LaneDetector(...).
ROI_TOP = 0.60
ROI_BOTTOM = 0.93
ROI_TOP_X = (0.32, 0.73)       # left / right x of the ROI top edge
ROI_BOTTOM_X = (0.03, 0.98)    # left / right x of the ROI bottom edge

# Pixel values below are tuned for a 640-pixel-wide frame and are
# scaled automatically for other widths.
REFERENCE_WIDTH = 640

# Marking pixels
WHITE_MIN_LIGHTNESS = 185      # bright white, on a normal road
WHITE_ABOVE_ROAD = 60          # ... and at least this far above the road
FAINT_MIN_LIGHTNESS = 95       # faint white (local contrast) ...
FAINT_ABOVE_ROAD = 30          # ... and at least this far above the road
YELLOW_HUE = (17, 34)          # OpenCV hue range of yellow paint
YELLOW_MIN_SATURATION = 70     # dirt is below/above this hue band and
YELLOW_MIN_VALUE = 80          # sunlit foliage above it
MAX_MASK_FILL = 0.30           # above this the frame is unreadable
GLARE_ROAD_LEVEL = 210         # median road lightness that means glare

# Line candidates
MAX_RUN_FRACTION = 0.12        # wider horizontal runs are not markings
HOUGH_THRESHOLD = 10           # roughly "rows of marking along the line"
MAX_HOUGH_LINES = 100
MIN_ABS_SLOPE = 0.45           # |dx/dy| near the car; steeper is not
MAX_ABS_SLOPE = 5.0            # a boundary of the lane we are in
BAND_TOP_PX = 3.0              # inlier band half-width at ROI top
BAND_BOTTOM_PX = 10.0          # ... and at ROI bottom (perspective)
SUPPORT_BINS = 16              # vertical bins used to measure coverage
MIN_BINS = 4                   # dashed lines need only a few bins
MIN_SPAN = 0.35                # fraction of ROI height the support covers
DUPLICATE_OVERLAP = 0.5        # shared-point fraction that marks a duplicate
MAX_INNER_CLUTTER = 0.45       # marking points just inside the lane, per
                               # supporting point; higher means texture
MAX_SIDE_CANDIDATES = 4

# Curves: each line is grown upward from its near-field part.
ALLOW_CURVE = True
GROW_STEP = 0.12               # fraction of ROI height per growth step
GROW_BAND_SCALE = 1.5
CURVE_MIN_SPAN = 0.35          # support needed before fitting a curve
CURVE_MAX_DEVIATION = 0.12     # max bend from the chord, fraction of width

# Pair checks (fractions of frame width / height)
LANE_WIDTH_RANGE = (0.28, 0.80)   # ego-lane width at ROI bottom
LANE_WIDTH_TOLERANCE = 0.40       # hard limit once a width is learned
LANE_WIDTH_COST = 2.0             # soft preference for the learned width
LANE_WIDTH_LOCK_FRAMES = 5
LANE_WIDTH_RESET_FRAMES = 45
MIN_TOP_WIDTH = 0.03              # area starts where the lane is this wide
MIN_SHARED_SPAN = 0.12
VANISHING_Y_MIN = 0.30
VANISHING_Y_ABOVE_ROI = 0.05      # VP may sit this far below the ROI top
VANISHING_X_RANGE = (0.10, 0.90)

# Tracking
SMOOTHING = 0.5                # weight of the new measurement
JUMP_FRACTION = 0.05           # larger lateral jumps need confirmation
JUMP_CONFIRM_FRAMES = 2
HOLD_FRAMES = 4


# ------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------

def put_text(image, message, position, color=WHITE, scale=0.45):
    cv2.putText(
        image,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def make_polyline(fit, h, w, y_min=None, y_max=None):
    if fit is None:
        return None

    start = fit["y_min"] if y_min is None else y_min
    stop = fit["y_max"] if y_max is None else y_max

    if stop <= start:
        return None

    ys = np.linspace(start, stop, 80)
    xs = np.polyval(fit["coefficients"], ys)

    return np.round(
        np.column_stack((xs * w, ys * h))
    ).astype(np.int32)


def draw_lane_area(frame, left, right, valid=True):
    output = frame.copy()

    if valid and left is not None and right is not None:
        polygon = np.vstack((left, right[::-1]))

        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (180, 180, 0))

        output = cv2.addWeighted(
            overlay, 0.22, output, 0.78, 0
        )

    # A missing side does not hide the other accepted side.
    for points, color in ((left, BLUE), (right, GREEN)):
        if points is not None:
            cv2.polylines(
                output,
                [points],
                False,
                color,
                2,
                cv2.LINE_AA,
            )

    return output


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def x_at(fit, y):
    """Normalized x of a fit at normalized y (scalar or array)."""
    return np.polyval(fit["coefficients"], y)


def chord(fit, start, stop):
    """Straight line (slope, intercept) through a fit's two end points."""
    x0, x1 = x_at(fit, start), x_at(fit, stop)
    slope = (x1 - x0) / (stop - start)
    return slope, x0 - slope * start


def fit_pixels(xs, ys, curved):
    """Quadratic (or straight) x = f(y) in pixels, highest power first."""
    if curved:
        return np.polyfit(ys, xs, 2)

    slope, intercept = np.polyfit(ys, xs, 1)
    return np.array([0.0, slope, intercept])


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

class LaneDetector:
    def __init__(
        self,
        roi_top=ROI_TOP,
        roi_bottom=ROI_BOTTOM,
        roi_top_x=ROI_TOP_X,
        roi_bottom_x=ROI_BOTTOM_X,
    ):
        if not 0.0 <= roi_top < roi_bottom <= 1.0:
            raise ValueError("Need 0 <= roi_top < roi_bottom <= 1.")

        self.top = float(roi_top)
        self.bottom = float(roi_bottom)
        self.top_x = tuple(roi_top_x)
        self.bottom_x = tuple(roi_bottom_x)
        self.reset()

    def reset(self):
        self.previous = {"left": None, "right": None}
        self.misses = {"left": 0, "right": 0}
        self.jumps = {"left": 0, "right": 0}
        self.lane_width = None
        self.width_frames = 0
        self.frames_without_pair = 0
        self.last_mask = None
        self.last_debug = None
        self.last_candidates = []
        self.last_fill = 0.0
        self.last_road_level = 0.0
        self.frame_shape = None

    # -------------------- region of interest --------------------

    def roi_polygon(self, h, w):
        return np.array([
            [int(w * self.bottom_x[0]), int(h * self.bottom)],
            [int(w * self.top_x[0]), int(h * self.top)],
            [int(w * self.top_x[1]), int(h * self.top)],
            [int(w * self.bottom_x[1]), int(h * self.bottom)],
        ], dtype=np.int32)

    def roi_mask(self, h, w):
        roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi, [self.roi_polygon(h, w)], 255)
        return roi

    # -------------------- marking pixels --------------------

    def marking_mask(self, frame):
        """Yellow and white candidates, adapted to road brightness."""
        h, w = frame.shape[:2]
        roi = self.roi_mask(h, w)

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lightness = hls[:, :, 1]
        saturation = hls[:, :, 2]

        road = lightness[roi > 0]
        road_level = float(np.median(road)) if road.size else 0.0
        self.last_road_level = road_level

        yellow = cv2.inRange(
            hsv,
            np.array([YELLOW_HUE[0], YELLOW_MIN_SATURATION,
                      YELLOW_MIN_VALUE], dtype=np.uint8),
            np.array([YELLOW_HUE[1], 255, 255], dtype=np.uint8),
        )

        bright_white = (
            (lightness >= max(WHITE_MIN_LIGHTNESS,
                              road_level + WHITE_ABOVE_ROAD))
            & (saturation <= 125)
        )

        kernel_width = max(15, int(w * 0.045))
        if kernel_width % 2 == 0:
            kernel_width += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, 3)
        )

        local_contrast = cv2.morphologyEx(
            lightness, cv2.MORPH_TOPHAT, kernel
        )

        faint_white = (
            (local_contrast >= 12)
            & (lightness >= max(FAINT_MIN_LIGHTNESS,
                                road_level + FAINT_ABOVE_ROAD))
            & (saturation <= 125)
        )

        white = (
            (bright_white | faint_white).astype(np.uint8) * 255
        )

        mask = cv2.bitwise_or(white, yellow)
        mask = cv2.bitwise_and(mask, roi)

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
        )

        keep = np.zeros(count, dtype=bool)
        min_area = max(3, int(h * w * 0.000015))

        for component in range(1, count):
            _, _, bw, bh, area = stats[component]

            if area < min_area:
                continue

            # Reject strongly horizontal components.
            if bw > 5 * bh and bw > w * 0.08:
                continue

            keep[component] = True

        cleaned = np.where(keep[labels], 255, 0).astype(np.uint8)

        roi_pixels = max(1, np.count_nonzero(roi))
        self.last_fill = np.count_nonzero(cleaned) / roi_pixels

        return cleaned

    def centerline_points(self, mask):
        """One point per horizontal run of marking pixels.

        Thick near-field dashes and thin far-field markings then carry
        similar weight per image row, and a line through a single blob
        at the wrong angle collects few votes. Runs cut off by the ROI
        sides are dropped: a large bright area clipped by the ROI edge
        would otherwise leave a fake line along that edge.
        """
        h, w = mask.shape
        padded = np.pad(mask > 0, ((0, 0), (1, 1)))
        steps = np.diff(padded.astype(np.int8), axis=1)

        start_y, start_x = np.nonzero(steps == 1)
        _, end_x = np.nonzero(steps == -1)

        lengths = end_x - start_x
        ok = lengths <= max(4, int(w * MAX_RUN_FRACTION))

        t = np.clip(
            (start_y / h - self.top) / (self.bottom - self.top), 0.0, 1.0
        )
        roi_left = w * (self.top_x[0] + (self.bottom_x[0] - self.top_x[0]) * t)
        roi_right = w * (self.top_x[1] + (self.bottom_x[1] - self.top_x[1]) * t)
        ok &= (start_x > roi_left + 1) & (end_x - 1 < roi_right - 1)

        xs = (start_x[ok] + end_x[ok] - 1) / 2.0
        ys = start_y[ok].astype(np.float64)
        return xs, ys

    # -------------------- line candidates --------------------

    def band(self, ys, h, w):
        y_top, y_bottom = self.top * h, self.bottom * h
        t = np.clip((ys - y_top) / (y_bottom - y_top), 0.0, 1.0)
        scale = w / REFERENCE_WIDTH
        return (BAND_TOP_PX + (BAND_BOTTOM_PX - BAND_TOP_PX) * t) * scale

    def coverage(self, ys, h):
        y_top, y_bottom = self.top * h, self.bottom * h
        height = y_bottom - y_top
        bins = np.floor((ys - y_top) / height * SUPPORT_BINS)
        bins = np.unique(np.clip(bins, 0, SUPPORT_BINS - 1))
        span = (np.max(ys) - np.min(ys)) / height
        return len(bins), span

    def refine_straight(self, xs, ys, slope, intercept, h, w):
        """Least-squares straight line inside a perspective-scaled band."""
        for _ in range(3):
            inliers = (
                np.abs(xs - (slope * ys + intercept))
                <= self.band(ys, h, w)
            )

            if np.count_nonzero(inliers) < 6:
                return None

            slope, intercept = np.polyfit(ys[inliers], xs[inliers], 1)

            if abs(slope) > MAX_ABS_SLOPE:
                return None

        inliers = (
            np.abs(xs - (slope * ys + intercept)) <= self.band(ys, h, w)
        )

        if np.count_nonzero(inliers) < 6:
            return None

        return np.array([0.0, slope, intercept]), inliers

    def grow_curve(self, xs, ys, slope, intercept, h, w):
        """Follow a marking upward from its near-field part.

        Each step refits to the points found so far and extends the
        search a little higher along that fit, so a bending marking
        is followed instead of being cut off by a straight line.
        """
        y_top, y_bottom = self.top * h, self.bottom * h
        height = y_bottom - y_top
        band = self.band(ys, h, w) * GROW_BAND_SCALE

        inliers = (
            (np.abs(xs - (slope * ys + intercept)) <= band)
            & (ys >= y_top + 0.45 * height)
        )

        if np.count_nonzero(inliers) < 6:
            return None

        reach = float(np.min(ys[inliers]))

        for _ in range(int(np.ceil(1.0 / GROW_STEP)) + 1):
            span = np.ptp(ys[inliers]) / height
            coefficients = fit_pixels(
                xs[inliers], ys[inliers],
                curved=span >= CURVE_MIN_SPAN
                and np.count_nonzero(inliers) >= 10,
            )

            reach = max(y_top, reach - GROW_STEP * height)
            grown = (
                (np.abs(xs - np.polyval(coefficients, ys)) <= band)
                & (ys >= reach)
            )

            # Stop if the extended fit lost much of what it had.
            if np.count_nonzero(grown) < 0.8 * np.count_nonzero(inliers):
                break

            inliers = grown

            if reach <= y_top:
                break

        span = np.ptp(ys[inliers]) / height
        curved = span >= CURVE_MIN_SPAN and np.count_nonzero(inliers) >= 10
        coefficients = fit_pixels(xs[inliers], ys[inliers], curved)

        final = (
            np.abs(xs - np.polyval(coefficients, ys)) <= self.band(ys, h, w)
        )

        if np.count_nonzero(final) < 6:
            return None

        coefficients = fit_pixels(xs[final], ys[final], curved)

        # Bounded bend: compare with the chord over the supported range.
        if curved:
            y0, y1 = float(np.min(ys[final])), y_bottom
            sample = np.linspace(y0, y1, 40)
            curve = np.polyval(coefficients, sample)
            line = np.interp(sample, [y0, y1], [curve[0], curve[-1]])

            if np.max(np.abs(curve - line)) > CURVE_MAX_DEVIATION * w:
                return None

        return coefficients, final

    def build_candidate(self, xs, ys, slope, intercept, h, w):
        straight = self.refine_straight(xs, ys, slope, intercept, h, w)

        if straight is None:
            return None

        coefficients, inliers = straight

        # A boundary of the lane the car is in leans toward the
        # vanishing point; a nearly vertical line in the middle of the
        # lane is texture, glare, or a vehicle edge.
        if abs(coefficients[1]) < MIN_ABS_SLOPE:
            return None

        if ALLOW_CURVE:
            grown = self.grow_curve(
                xs, ys, coefficients[1], coefficients[2], h, w
            )

            if grown is not None:
                g_coefficients, g_inliers = grown
                g_bins, _ = self.coverage(ys[g_inliers], h)
                s_bins, _ = self.coverage(ys[inliers], h)

                if (
                    g_bins >= s_bins
                    and np.count_nonzero(g_inliers)
                    >= np.count_nonzero(inliers)
                ):
                    coefficients, inliers = g_coefficients, g_inliers

        in_x, in_y = xs[inliers], ys[inliers]
        bins, span = self.coverage(in_y, h)

        if bins < MIN_BINS or span < MIN_SPAN:
            return None

        # Near-field lean must also hold after any curve fitting.
        y_bottom = self.bottom * h
        near_y = y_bottom - 0.3 * (y_bottom - self.top * h)
        near_slope = (
            np.polyval(coefficients, y_bottom)
            - np.polyval(coefficients, near_y)
        ) / (y_bottom - near_y)

        if not MIN_ABS_SLOPE <= abs(near_slope) <= MAX_ABS_SLOPE:
            return None

        # The lane side of a real boundary is plain road. Measure marking
        # points just inside it (lower ROI only, where lines are apart).
        t = (ys - self.top * h) / (y_bottom - self.top * h)
        lower = t > 0.3
        offset = xs - np.polyval(coefficients, ys)
        inward = -offset if near_slope > 0 else offset
        band = self.band(ys, h, w)
        inner = lower & (inward > band) & (inward <= 4 * band)
        supporting = np.count_nonzero(inliers & lower)

        if (
            supporting >= 5
            and np.count_nonzero(inner) > MAX_INNER_CLUTTER * supporting
        ):
            return None

        y_min = max(self.top, float(np.min(in_y)) / h)
        sample = np.polyval(
            coefficients, np.linspace(y_min * h, y_bottom, 30)
        )

        if np.any(sample < -0.25 * w) or np.any(sample > 1.25 * w):
            return None

        error_px = float(np.mean(np.abs(
            in_x - np.polyval(coefficients, in_y)
        )))

        a, b, c = coefficients

        return {
            "coefficients": np.array([a * h * h / w, b * h / w, c / w]),
            "y_min": y_min,
            "y_max": self.bottom,
            "observed_y_max": float(np.max(in_y)) / h,
            "support": int(bins),
            "span": float(span),
            "count": int(len(in_y)),
            "error": error_px / w,
            "curved": bool(a != 0.0),
            "centers": np.round(
                np.column_stack((in_x, in_y))
            ).astype(np.int32),
            "_inliers": inliers,
        }

    def find_candidates(self, mask):
        h, w = mask.shape
        xs, ys = self.centerline_points(mask)

        if len(xs) < 6:
            return []

        centerline = np.zeros_like(mask)
        centerline[ys.astype(int), np.round(xs).astype(int)] = 255

        lines = cv2.HoughLines(
            centerline, 1, np.pi / 360, HOUGH_THRESHOLD
        )

        if lines is None:
            return []

        candidates = []

        for rho, theta in lines[:MAX_HOUGH_LINES, 0]:
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            if abs(cos_t) < 1e-3:
                continue

            # x * cos + y * sin = rho  ->  x = slope * y + intercept
            slope = -sin_t / cos_t
            intercept = rho / cos_t

            if not MIN_ABS_SLOPE * 0.5 <= abs(slope) <= MAX_ABS_SLOPE:
                continue

            fit = self.build_candidate(xs, ys, slope, intercept, h, w)

            if fit is not None:
                candidates.append(fit)

        # Best-supported first. Several Hough peaks usually describe the
        # same marking at slightly different angles; any candidate that
        # mostly reuses the points of a better one is a duplicate.
        candidates.sort(
            key=lambda f: (f["support"], f["count"], -f["error"]),
            reverse=True,
        )

        unique = []

        for fit in candidates:
            mine = fit["_inliers"]
            duplicate = any(
                np.count_nonzero(mine & kept["_inliers"])
                >= DUPLICATE_OVERLAP * min(
                    np.count_nonzero(mine),
                    np.count_nonzero(kept["_inliers"]),
                )
                for kept in unique
            )

            if not duplicate:
                unique.append(fit)

        for fit in unique:
            del fit["_inliers"]

        return unique

    def near_slope(self, fit):
        """Normalized dx/dy over the lower part of the ROI."""
        y1 = self.bottom
        y0 = self.bottom - 0.3 * (self.bottom - self.top)
        return (x_at(fit, y1) - x_at(fit, y0)) / (y1 - y0)

    def split_sides(self, candidates):
        """Lines left of the vanishing point lean one way, right the other."""
        left, right = [], []

        for fit in candidates:
            (left if self.near_slope(fit) < 0 else right).append(fit)

        # Innermost (closest to the vehicle) first.
        left.sort(key=lambda f: x_at(f, self.bottom), reverse=True)
        right.sort(key=lambda f: x_at(f, self.bottom))

        return left[:MAX_SIDE_CANDIDATES], right[:MAX_SIDE_CANDIDATES]

    def lateral_shift(self, a, b):
        """Largest normalized x difference over the shared range."""
        top = max(a["y_min"], b["y_min"])
        ys = np.array([top, 0.5 * (top + self.bottom), self.bottom])
        return float(np.max(np.abs(x_at(a, ys) - x_at(b, ys))))

    # -------------------- pair logic --------------------

    def check_pair(self, left, right):
        """Geometric checks shared by pair selection and validation."""
        if left is None and right is None:
            return False, "Neither boundary accepted", None

        if left is None:
            return False, "Left boundary missing", None

        if right is None:
            return False, "Right boundary missing", None

        start = max(left["y_min"], right["y_min"])
        stop = min(left["y_max"], right["y_max"])

        if stop - start < MIN_SHARED_SPAN:
            return False, "Insufficient shared support", None

        ys = np.linspace(start, stop, 80)
        widths = x_at(right, ys) - x_at(left, ys)

        if not np.all(np.isfinite(widths)):
            return False, "Invalid boundary values", None

        # Boundaries converge toward the vanishing point, which can sit
        # right at the ROI top. Start the area where it is wide enough
        # and require it to stay that wide all the way down.
        wide = widths >= MIN_TOP_WIDTH

        if not wide[-1]:
            return False, "Crossing or out-of-image boundaries", None

        first = len(wide) - int(np.argmin(wide[::-1])) if not wide.all() else 0
        start = float(ys[first])

        if stop - start < MIN_SHARED_SPAN:
            return False, "Region becomes too narrow", None

        ys = ys[first:]
        lx = x_at(left, ys)
        rx = x_at(right, ys)
        widths = rx - lx

        if np.any(lx < -0.15) or np.any(rx > 1.15):
            return False, "Crossing or out-of-image boundaries", None

        bottom_width = widths[-1]
        low, high = LANE_WIDTH_RANGE

        if not low <= bottom_width <= high:
            return False, "Lower lane-width check failed", None

        if (
            self.lane_width is not None
            and self.width_frames >= LANE_WIDTH_LOCK_FRAMES
            and abs(bottom_width - self.lane_width)
            > LANE_WIDTH_TOLERANCE * self.lane_width
        ):
            return False, "Width differs from learned lane width", None

        if bottom_width < widths[0] * 1.25:
            return False, "Perspective width check failed", None

        # Vanishing point from the near-field direction of each side
        # (on a curve the far ends bend away; the near field does not).
        near_start = start + 0.4 * (stop - start)
        ls, li = chord(left, near_start, stop)
        rs, ri = chord(right, near_start, stop)

        if abs(rs - ls) < 1e-6:
            return False, "Boundaries do not converge", None

        vy = (li - ri) / (rs - ls)
        vx = ls * vy + li

        if not (
            VANISHING_Y_MIN <= vy <= self.top + VANISHING_Y_ABOVE_ROI
            and VANISHING_X_RANGE[0] <= vx <= VANISHING_X_RANGE[1]
        ):
            return False, "Implausible vanishing point", None

        return True, "Boundary pair accepted", (start, stop)

    def select_pair(self, lefts, rights):
        """Prefer the innermost valid pair, then continuity, then support."""
        options = []

        for i, left in enumerate(lefts):
            for j, right in enumerate(rights):
                ok, _, _ = self.check_pair(left, right)

                if not ok:
                    continue

                cost = float(i + j)

                if (
                    self.lane_width is not None
                    and self.width_frames >= LANE_WIDTH_LOCK_FRAMES
                ):
                    width = x_at(right, self.bottom) - x_at(left, self.bottom)
                    cost += LANE_WIDTH_COST * abs(
                        width - self.lane_width
                    ) / self.lane_width

                for fit, side in ((left, "left"), (right, "right")):
                    previous = self.previous[side]

                    if previous is not None:
                        shift = self.lateral_shift(fit, previous)
                        cost += 0.5 * min(shift / JUMP_FRACTION, 2.0)

                support = left["support"] + right["support"]
                options.append((cost, -support, i, j))

        if not options:
            return None, None

        _, _, i, j = min(options)
        return lefts[i], rights[j]

    def select_single(self, candidates, side):
        """Used when no valid pair exists in this frame."""
        if not candidates:
            return None

        # Once a lane width is learned, a lone line must still form a
        # plausible lane with the other side's last boundary. This stops
        # an outer edge line from taking over when a dash is missing.
        other = self.previous["right" if side == "left" else "left"]

        if other is not None and self.width_frames >= LANE_WIDTH_LOCK_FRAMES:
            candidates = [
                fit for fit in candidates
                if self.check_pair(
                    *((fit, other) if side == "left" else (other, fit))
                )[0]
            ]

            if not candidates:
                return None

        previous = self.previous[side]

        if previous is not None:
            return min(
                candidates, key=lambda f: self.lateral_shift(f, previous)
            )

        return candidates[0]

    # -------------------- tracking --------------------

    def track(self, fit, side):
        """Smooth, confirm large jumps, and hold briefly on misses.

        Returns (fit or None, held).
        """
        previous = self.previous[side]

        trusted = previous is not None and previous.get("paired", False)

        if fit is None:
            self.misses[side] += 1
            self.jumps[side] = 0

            if trusted and self.misses[side] <= HOLD_FRAMES:
                return dict(previous), True

            self.previous[side] = None
            return None, False

        if trusted:
            if self.lateral_shift(fit, previous) > JUMP_FRACTION:
                self.jumps[side] += 1

                if self.jumps[side] < JUMP_CONFIRM_FRAMES:
                    self.misses[side] += 1

                    if self.misses[side] <= HOLD_FRAMES:
                        return dict(previous), True

                    self.previous[side] = None
                    return None, False
            else:
                fit = dict(fit)
                fit["coefficients"] = (
                    SMOOTHING * fit["coefficients"]
                    + (1.0 - SMOOTHING) * previous["coefficients"]
                )

        self.misses[side] = 0
        self.jumps[side] = 0
        self.previous[side] = dict(fit, paired=False)
        return fit, False

    def update_lane_width(self, left, right, valid):
        if valid:
            width = float(x_at(right, self.bottom) - x_at(left, self.bottom))
            self.frames_without_pair = 0
            self.lane_width = (
                width if self.lane_width is None
                else 0.9 * self.lane_width + 0.1 * width
            )
            self.width_frames += 1
            return

        self.frames_without_pair += 1

        # Do not let a stale width lock the detector out forever.
        if self.frames_without_pair >= LANE_WIDTH_RESET_FRAMES:
            self.lane_width = None
            self.width_frames = 0

    # -------------------- main entry --------------------

    def detect(self, frame):
        h, w = frame.shape[:2]

        if self.frame_shape != (h, w):
            self.reset()
            self.frame_shape = (h, w)

        mask = self.marking_mask(frame)
        self.last_mask = mask

        unreadable = (
            self.last_fill > MAX_MASK_FILL
            or self.last_road_level > GLARE_ROAD_LEVEL
        )
        candidates = [] if unreadable else self.find_candidates(mask)
        self.last_candidates = candidates
        lefts, rights = self.split_sides(candidates)

        measured_left, measured_right = self.select_pair(lefts, rights)

        if measured_left is None:
            measured_left = self.select_single(lefts, "left")
            measured_right = self.select_single(rights, "right")

        left, left_held = self.track(measured_left, "left")
        right, right_held = self.track(measured_right, "right")

        valid, reason, shared = self.check_pair(left, right)
        self.update_lane_width(left, right, valid)

        # Only boundaries of an accepted lane may be held later.
        for side, held in (("left", left_held), ("right", right_held)):
            if self.previous[side] is not None and not held:
                self.previous[side]["paired"] = valid

        if unreadable:
            reason = (
                "Markings unreadable (glare or heavy texture)"
                if not valid else reason
            )

        held = [s for s, f in (("left", left_held), ("right", right_held)) if f]
        if held:
            reason += f" (holding {' + '.join(held)})"

        if valid:
            start, stop = shared
            left_points = make_polyline(left, h, w, start, stop)
            right_points = make_polyline(right, h, w, start, stop)
        else:
            left_points = make_polyline(left, h, w)
            right_points = make_polyline(right, h, w)

        # Unaccepted lines stay in the debug view and the return values,
        # but are not drawn on the video, where they would look trusted.
        if valid:
            annotated = draw_lane_area(
                frame, left_points, right_points, True
            )
        else:
            annotated = frame.copy()

        self.last_debug = self.draw_debug(
            mask, candidates, left, right, left_points, right_points
        )

        info = {
            "left_detected": left is not None,
            "right_detected": right is not None,
            "lane_area_valid": valid,
            "reason": reason,
            "left_support": left["support"] if left else 0,
            "right_support": right["support"] if right else 0,
            "left_possible": None,
            "right_possible": None,
            # Extra diagnostics; existing keys are unchanged.
            "left_held": left_held,
            "right_held": right_held,
            "left_candidates": len(lefts),
            "right_candidates": len(rights),
            "mask_fill": round(self.last_fill, 3),
        }

        return annotated, left_points, right_points, info

    def draw_debug(self, mask, candidates, left, right,
                   left_points, right_points):
        h, w = mask.shape
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.polylines(
            debug, [self.roi_polygon(h, w)], True, DARK_GRAY, 1
        )

        for fit in candidates:
            points = make_polyline(fit, h, w)
            if points is not None:
                cv2.polylines(
                    debug, [points], False, DIM_GRAY, 1, cv2.LINE_AA
                )

        for fit, points, color in (
            (left, left_points, BLUE),
            (right, right_points, GREEN),
        ):
            if fit is not None:
                for x, y in fit["centers"]:
                    cv2.circle(debug, (int(x), int(y)), 2, color, -1)

            if points is not None:
                cv2.polylines(
                    debug, [points], False, color, 2, cv2.LINE_AA
                )

        return debug


# Compatible with the latest curved-boundary Phase 3.
_default_detector = LaneDetector()


def detect_lanes(frame):
    return _default_detector.detect(frame)


# ------------------------------------------------------------
# Standalone playback
# ------------------------------------------------------------

def window_open():
    try:
        return cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE
        ) >= 1
    except cv2.error:
        return False


def build_display(annotated, debug, info, index, paused):
    combined = np.hstack((annotated, debug))
    h, w = combined.shape[:2]

    display = cv2.copyMakeBorder(
        combined, 38, 140, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )

    put_text(display, "VIDEO + ESTIMATED LANE", (15, 25))
    put_text(
        display,
        "MASK + CANDIDATES (GRAY) + CHOSEN BOUNDARIES",
        (w // 2 + 15, 25),
    )

    state = "PAUSED" if paused else "PLAYING"
    area = "DISPLAYED" if info["lane_area_valid"] else "HIDDEN"

    def side_text(side):
        text = f"{info[side + '_support']}/{SUPPORT_BINS}"
        return text + (" held" if info.get(side + "_held") else "")

    rows = [
        (
            f"Lane area {area}: {info['reason']}",
            YELLOW,
        ),
        (
            f"Support bins: LEFT {side_text('left')} | "
            f"RIGHT {side_text('right')}",
            WHITE,
        ),
        (
            f"Line candidates: LEFT {info.get('left_candidates', 0)} | "
            f"RIGHT {info.get('right_candidates', 0)}",
            WHITE,
        ),
        (
            f"Frame {index} | {state} | Space: pause | "
            "N: step | R: restart | Q: quit",
            WHITE,
        ),
        (
            "Shading is an estimate. Check alignment with actual markings.",
            GRAY,
        ),
    ]

    for row, (message, color) in enumerate(rows):
        put_text(
            display, message,
            (15, h + 62 + row * 24),
            color,
        )

    return display


def run_phase1(video_path):
    video_path = Path(video_path).expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            "Place lane_video.mp4 beside this Python file "
            "or pass its full path when running."
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    detector = LaneDetector()
    paused = False
    advance = True
    index = -1

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 600)

        print(f"Video: {video_path}", flush=True)
        print("Click the video window for controls.")
        print("Space: pause/resume | N: step | R: restart | Q: quit")

        while True:
            if advance:
                ret, frame = cap.read()

                if not ret:
                    print("End of video or no more readable frames.")
                    break

                index += 1
                h, w = frame.shape[:2]

                frame = cv2.resize(
                    frame,
                    (
                        DISPLAY_WIDTH,
                        max(1, round(h * DISPLAY_WIDTH / w)),
                    ),
                )

                annotated, _, _, info = detector.detect(frame)

                if index % 30 == 0:
                    print(
                        f"[{index:05d}] "
                        f"Left: {info['left_detected']} | "
                        f"Right: {info['right_detected']} | "
                        f"Area: {info['lane_area_valid']} | "
                        f"{info['reason']}",
                        flush=True,
                    )

            display = build_display(
                annotated,
                detector.last_debug,
                info,
                index,
                paused,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(
                30 if paused else PLAYBACK_DELAY_MS
            ) & 0xFF

            if key in QUIT_KEYS or not window_open():
                break

            if key in (ord("r"), ord("R")):
                cap.release()
                cap = cv2.VideoCapture(str(video_path))

                if not cap.isOpened():
                    raise RuntimeError("Could not restart the video.")

                detector.reset()
                index = -1
                advance = True
                continue

            if key == ord(" "):
                paused = not paused

            advance = not paused

            if paused and key in (ord("n"), ord("N")):
                advance = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 lane-area diagnostics."
    )

    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Optional full path to a video.",
    )

    args = parser.parse_args()
    run_phase1(args.video)