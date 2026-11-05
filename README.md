# Dashcam Driver-Assistance Prototype

A Python prototype that processes dashcam video in three phases. It finds the boundaries of the lane the car is in using classical computer vision, detects nearby road users with a pretrained YOLO model, and combines both into one prioritized status message. The focus is on measuring where each part fails and reporting uncertainty instead of guessing.

> Research prototype only. It is not a safety system, distances are uncalibrated estimates, and it never decides whether a lane change is safe.

## Pipeline

| Phase | File | What it does |
|---|---|---|
| 1. Lane boundaries | `phase1_lane_detection.py` | Classical computer vision (OpenCV): finds lane-marking pixels, fits the left and right boundaries of the current lane, accepts the lane only when geometry checks pass, and draws a smoothed lane |
| 2. Objects | `phase2_object_detection.py` | Pretrained YOLO (COCO classes) with ByteTrack tracking and an approximate distance for each object |
| 3. Decision | `phase3_decision_system.py` | Combines Phases 1 and 2: places each object relative to the detected lane and produces one prioritized status per frame |
| Evaluation | `evaluate_pipeline.py`, `evaluate_phase3.py` | Run the phases without a display and report acceptance rates, failure reasons, accuracy checks, tracking losses, alert rates and timing |

### Phase 1: lane boundaries

- **Marking pixels:** white and yellow paint, with thresholds that adapt to the road's brightness so sunlit or washed-out asphalt does not flood the mask. Frames dominated by glare or texture are reported as unreadable instead of guessed.
- **Line candidates:** Hough voting on the centerline of each marking segment, so gaps between dashes do not matter. Each line is then grown upward to follow curves.
- **Choosing the lane:** the innermost left/right pair that passes geometry checks: a plausible lane width and vanishing point (both learned over time for each video), narrowing with perspective, and clean road on the lane side of each line.
- **Tracking:** smoothing, confirmation before accepting a large jump, and holding a briefly missing boundary for at most 4 frames. Held boundaries are labeled as held, and nothing is drawn when the lane is not accepted.
- **Smoothed display:** each accepted boundary is sampled at fixed image rows and blended with previous frames, faster when the lane keeps moving. If the lane is hidden for 1-3 frames, the last lane stays on screen, fainter and labeled as held. `detect()` itself returns the raw boundaries, so every evaluation measures the detector, not the smoothing.

### Phase 2: objects

YOLO detects people, vehicles and animals, and ByteTrack keeps track IDs so distance estimates can be smoothed over time. Distance comes from a pinhole-camera estimate using an assumed real-world width per object class and an uncalibrated focal length, so it is approximate.

### Phase 3: decision

- **Lane-based zones:** an object is "front" when the point where it touches the road (the bottom-center of its box) lies inside the accepted lane, and "left" or "right" when it lies outside. Objects beyond the lane area, and frames without an accepted lane, fall back to fixed image slices (the middle 35%-65% of the width is "front").
- **One status per frame, in priority order:** FRONT OBJECT NEAR (an object in front within an estimated 15 m), BOUNDARY DETECTION INCOMPLETE (no accepted lane), then FRONT OBJECT DETECTED or NO FRONT OBJECT DETECTED.
- A filter removes detections of the camera car's own hood, which YOLO sometimes labeled as a car about 1.5 m ahead.
- Lane-change availability is always reported as unknown.

## Test videos

| Level | Video | Frames | What makes it hard |
|---|---|---|---|
| Easy | lane_video.mp4 | 295 | Straight highway, clear markings, heavy traffic in other lanes |
| Medium | random.mp4 | 503 | Rural road, curves, grass along the edge, yellow edge line |
| Hard | harder_challenge_video.mp4 | 1,199 | Mountain road, sharp curves, deep shadows, glare, motorcycles; the car's hood is in view, so the lane search stops at 90% of the frame height |

## Results

Measured on a MacBook Air (Apple M5), Python 3.11, OpenCV 5.0, with YOLO11n, using `evaluate_pipeline.py` and `evaluate_phase3.py`. Raw outputs and result images are in `results/`.

### Phase 1: lane detection (640-pixel frames)

| | Easy | Medium | Hard |
|---|---|---|---|
| Frames with an accepted lane | 294 (99.7%) | 502 (99.8%) | 885 (73.8%) |
| Longest run with no lane shown | 1 frame | 1 frame | 66 frames |
| Frames reported as unreadable (glare or texture) | 0 | 0 | 84 |
| Accepted frames with the left boundary on the yellow line* | n/a (white line) | 502 of 502 (100%) | 707 of 839 (84.3%)** |
| Lane detection time (median) | 11.5 ms | 14.5 ms | 25.9 ms |

\* An automated accuracy proxy: does the accepted left boundary sit on yellow paint? It covers the left boundary only.
\** Not yet checked by eye; deep shadows can also fool the check itself.

### Phases 2 and 3: objects, tracking and alerts (960-pixel frames)

| | Easy | Medium | Hard |
|---|---|---|---|
| Objects per frame (after the hood filter) | 5.46 | 0.07 | 0.30 |
| Objects YOLO detected but tracking dropped | 52 of 1,729 (3%) | 7 of 43 (16%) | 102 of 474 (22%) |
| ... of those, in front and within 15 m | 0 | 0 | 8 |
| Objects placed by the lane (the rest use fixed slices) | 19% | 100% | 54% |
| FRONT OBJECT NEAR frames: fixed slices -> lane-based | 0 -> 0 | 0 -> 0 | 48 -> 37 |
| Phase 3 processing time (lane + YOLO, medians) | ~41 ms (~24 FPS) | ~46 ms (~22 FPS) | ~57 ms (~18 FPS) |

### Lane smoothing (display)

| Raw -> smoothed | Easy | Medium | Hard |
|---|---|---|---|
| Frames where part of a line jumped more than 1.5% of the width | 4.1% -> 0.0% | 3.6% -> 0.0% | 10.6% -> 4.5% |
| Frames where the line's top end jumped more than 1.5% of the width | 4.8% -> 0.0% | 0.6% -> 0.0% | 14.3% -> 0.8% |
| Times the lane switched between shown and hidden | 2 -> 0 | 2 -> 0 | 50 -> 24 |

Measured on Phase 1's 640-pixel output during development; this check is not yet part of the evaluation scripts. The cost is a small lag: on the hard video the smoothed line trails the measurement by a median of about 0.5% of the frame width.

### What the results show

- **Lane detection degrades with difficulty, and says so.** It is near-perfect on the easy and medium roads. On the hard road it shows a lane in 74% of frames and marks 84 frames as unreadable instead of guessing.
- **Lane-based zones matter on curves.** On the hard video the two zone rules disagreed about the alert in 21 frames. In the 12 saved for review (`results/phase3_disagreements_harder_challenge_video.jpg`), the lane rule was right every time: it removed 9 false alarms (oncoming vehicles in the other lane that sat in the middle of the image on a curve) and added 3 correct ones (a motorcycle just ahead at the left edge of the lane). On the straight highway the rules never disagreed.
- **Tracking loses fast-moving traffic.** The share of detections that tracking drops rises from 3% on the highway to 22% on the mountain road, mostly vehicles passing quickly or at the edge of the frame. It also filters some false detections (an orange roadside post labeled "person"), so simply using raw detections is not the fix.
- **Only the hard video exercises the alert logic.** The easy and medium videos never have an object in front within 15 m, so they cannot test the alerts.

## From revision 4 to revision 5 (Phase 1)

| Medium video | Revision 4 | Revision 5 |
|---|---|---|
| Frames with an accepted lane | 329 (65.4%) | 502 (99.8%) |
| Accepted frames with the left boundary off the yellow line | 49 (14.9%) | 0 |
| Longest run with no lane shown | 28 | 1 |

Measuring revision 4 on the medium video showed that it often confidently chose the wrong left boundary. Tracing those frames found two causes:

1. **Grass passing as yellow paint.** Sunlit roadside grass falls inside the hue range used for yellow paint. The real yellow line was more than twice as bright as the road, while the grass was about as bright as the asphalt. White paint already had to be brighter than the road; revision 5 applies the same rule to yellow.
2. **A fixed horizon limit.** The vanishing-point check assumed one camera tilt. In this video the correct lane pair implied a horizon just past that limit, so the right answer was rejected while a wrong pair sometimes passed. Revision 5 learns the horizon height for each video, as it already did for lane width.

On the hard video revision 5 is not an improvement: frames with a lane went from 907 (75.6%) to 885 (73.8%), the longest gap from 47 to 66 frames, and the yellow-line check stayed about the same (84.7% -> 84.3%). The fixes trade a little coverage on the hard video for a large gain on the medium one. Likely causes, not yet confirmed: the stricter yellow rule may miss faded or shadowed paint when the road is bright, and the learned horizon rejected 4 frames on sharp curves.

## Limitations

- There are no hand-labeled ground-truth boundaries. Lane accuracy is an automated proxy, and alert decisions were checked by eye on a sample of frames.
- The revision 5 fixes were developed using the medium video, so its results are not a held-out test, and none of the three videos is fully unseen.
- Only the hard video contains objects in front within 15 m, so the alert logic was tested on one video.
- Tracking drops fast-passing vehicles. A planned fix is to also accept detections seen in at least 2 of the last 3 frames.
- Distances use assumed object widths and an uncalibrated focal length. Vehicles seen side-on look closer than they are, because the box includes the car's length.
- Phase 1 runs lane detection on 640-pixel frames and Phase 3 on 960-pixel frames, so their lane numbers differ slightly.

## Running it

```bash
conda create -n cv_project python=3.11
conda activate cv_project
pip install -r requirements.txt
```

Place the videos in the project folder (they are not included in this repository), then:

```bash
python phase1_lane_detection.py random.mp4                     # smoothed lane; add --no-smoothing for raw lines
python phase2_object_detection.py lane_video.mp4
python phase3_decision_system.py                                # hard video: smoothed lane + lane-based zones
python phase3_decision_system.py lane_video.mp4 --roi-bottom 0.93
python phase3_decision_system.py --fixed-zones --no-smoothing   # previous behavior, for comparison
```

Evaluation:

```bash
python evaluate_pipeline.py random.mp4 --yellow-check --model yolo11n.pt
python evaluate_phase3.py harder_challenge_video.mp4 --roi-bottom 0.90
```

Window controls: Space to pause, N to step one frame, Q to quit (M shows or hides the mask in Phase 3). The YOLO weights download automatically on first run.

## Data

Three short dashcam clips were used for local testing. They are not redistributed here, and I did not record their original sources.

## My contribution

Developed since April 2026; earlier revisions were not tracked in Git.

I built revisions 1-4 with help from AI coding assistants. I defined what each phase should do, tested every revision on dashcam footage with curves, shadows and sun glare, identified where it failed, and decided what each revision needed to fix.

Revision 5 of Phase 1, the lane smoothing, the lane-based zones in Phase 3, and both evaluation scripts were developed with AI assistance (Anthropic's Claude). I ran all evaluations on my own machine.
