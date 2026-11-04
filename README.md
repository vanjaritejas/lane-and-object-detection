# Dashcam Driver-Assistance Prototype

A Python prototype that processes dashcam video in three phases. It estimates the boundaries of the lane the car is in using classical computer vision, detects nearby road users with a pretrained YOLO model, and combines both into one prioritized status message. The focus is on measuring where the estimates fail and reporting uncertainty instead of guessing.

> Research prototype only. It is not a safety system, distances are uncalibrated estimates, and it never decides whether a lane change is safe.

## Pipeline

| Phase | File | What it does |
|---|---|---|
| 1. Lane boundaries | `phase1_lane_detection.py` | Classical computer vision (OpenCV): finds lane-marking pixels, fits the left and right boundaries of the current lane, and shows the lane area only when geometry checks pass |
| 2. Objects | `phase2_object_detection.py` | Pretrained YOLO (COCO classes) with ByteTrack tracking; assigns each object to a left, front or right image zone with an approximate distance |
| 3. Decision | `phase3_decision_system.py` | Combines Phases 1 and 2 into one prioritized status message per frame |
| Evaluation | `evaluate_pipeline.py` | Runs the pipeline without a display and reports acceptance rates, failure reasons, an accuracy check and timing |

### Phase 1: lane boundaries

- **Marking pixels:** white and yellow paint, with thresholds that adapt to the road's brightness so sunlit or washed-out asphalt does not flood the mask. Frames dominated by glare or texture are reported as unreadable instead of guessed.
- **Line candidates:** Hough voting on the centerline of each marking segment, so gaps between dashes do not matter. Each line is then grown upward to follow curves.
- **Choosing the lane:** the innermost left/right pair that passes geometry checks: a plausible lane width and vanishing point (both learned over time for each video), narrowing with perspective, and clean road on the lane side of each line.
- **Tracking:** smoothing, confirmation before accepting a large jump, and holding a briefly missing boundary for at most 4 frames. Held boundaries are labeled as held, and nothing is drawn when the lane is not accepted.

### Phase 2: objects

YOLO detects people, vehicles and animals, and ByteTrack keeps track IDs so distance estimates can be smoothed over time. Distance comes from a pinhole-camera estimate using an assumed real-world width per object class and an uncalibrated focal length, so it is approximate. The left, front and right zones are fixed slices of the image, not detected lanes.

### Phase 3: decision

One status per frame, in priority order:

1. **FRONT OBJECT NEAR:** an object in the front zone within an estimated 15 m
2. **BOUNDARY DETECTION INCOMPLETE:** no accepted lane in this frame
3. **FRONT OBJECT DETECTED** or **NO FRONT OBJECT DETECTED**

Lane-change availability is always reported as unknown. A filter removes detections of the camera car's own hood, which YOLO sometimes labeled as a car about 1.5 m ahead, triggering false "front object near" alerts.

## Results (Phase 1)

Measured with `evaluate_pipeline.py` on a MacBook Air (Apple M5), Python 3.11, OpenCV 5.0. Timing covers lane detection only, on frames resized to 640 pixels wide.

**random.mp4:** 503 frames of a rural road with curves, grass along the edge and a yellow left edge line.

| | Revision 4 | Revision 5 |
|---|---|---|
| Frames with an accepted lane area | 329 (65.4%) | 502 (99.8%) |
| Accepted frames with the left boundary off the yellow line | 49 (14.9%) | 0 |
| Longest run of frames with no lane shown | 28 | 1 |
| Accepted frames relying on a held boundary | 166 | 14 |
| Lane detection time per frame (median) | 17.4 ms | 12.2 ms |

**lane_video.mp4:** 295 frames of highway driving. Revision 5 accepts a lane area in 294 frames (99.7%), at a median of 9.7 ms per frame.

"Off the yellow line" is an automated check of whether the accepted left boundary sits on yellow paint. It covers the left boundary only; the right boundary was checked by eye on sampled frames.

## From revision 4 to revision 5

Measuring revision 4 on a second video showed that it often confidently chose the wrong left boundary. Tracing those frames found two causes:

1. **Grass passing as yellow paint.** Sunlit roadside grass falls inside the hue range used for yellow paint. The real yellow line was more than twice as bright as the road, while the grass was about as bright as the asphalt. White paint already had to be brighter than the road; revision 5 applies the same rule to yellow.
2. **A fixed horizon limit.** The vanishing-point check assumed one camera tilt. In this video the correct lane pair implied a horizon just past that limit, so the right answer was rejected while a wrong pair sometimes passed. Revision 5 learns the horizon height for each video, as it already did for lane width, and then requires new pairs to stay close to it.

## Limitations

- Only two test videos, and the revision 5 fixes were developed using random.mp4 itself, so these results are not a held-out test. The next step is evaluating on footage neither revision has seen.
- There are no hand-labeled ground-truth boundaries; the accuracy figure is an automated proxy.
- Phase 2 and 3 distances use assumed object widths and an uncalibrated focal length.
- The timing figures cover Phase 1 only; YOLO adds processing time that is not yet measured here.

## Running it

```bash
conda create -n cv_project python=3.11
conda activate cv_project
pip install -r requirements.txt
```

Place the videos in the project folder (they are not included in this repository), then:

```bash
python phase1_lane_detection.py lane_video.mp4     # lane diagnostics window
python phase2_object_detection.py lane_video.mp4   # object detection window
python phase3_decision_system.py                   # currently reads ~/Downloads/harder_challenge_video.mp4
python evaluate_pipeline.py random.mp4 --yellow-check
```

Window controls: Space to pause, N to step one frame, Q to quit. The YOLO weights (`yolov8n.pt`, `yolo11n.pt`) download automatically on first run.

## Data

Two short dashcam clips were used for local testing. They are not redistributed here, and I did not record their original sources.

## My contribution

Developed since April 2026; earlier revisions were not tracked in Git.

I built revisions 1-4 with help from AI coding assistants. I defined what each phase should do, tested every revision on dashcam footage with curves, shadows and sun glare, identified where it failed, and decided what each revision needed to fix.

Revision 5 of Phase 1 (the two fixes above) and `evaluate_pipeline.py` were developed with AI assistance (Anthropic's Claude). I ran and verified all results on my own machine.
