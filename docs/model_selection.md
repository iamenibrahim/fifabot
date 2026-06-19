# Model Selection

Model choices should be based on validation evidence, not folder names or newest
timestamps. The current live bot uses the best scored existing model for each
detector based on saved YOLO `results.csv` files.

## Current Detector Selection

| Component | Selected run | mAP50-95 | Reason |
| --- | --- | ---: | --- |
| Ball | `ball_v3` | `0.30088` | Highest scored ball detector among saved metrics. |
| Player | `player_detector_v1-5` | `0.32834` | Highest scored player detector with saved metrics. |
| Indicator | `indicator_v1` | `0.38524` | Only scored indicator detector. |

`player_detector_v1-6` has a `best.pt`, but no saved `results.csv` was found in
its run folder. It is not selected until it is validated and beats
`player_detector_v1-5`.

The selected ball weights are only part of the tracking behavior. Runtime
inference size also matters because the ball is very small. The live loop uses
`ball_v3` with `imgsz=928` and `conf=0.40`, matching the standalone ball tracker
settings that were more reliable than the original live-loop `imgsz=320`.

## Current Game-State Selection

The live bot uses the existing `fifa_state_classifier.pkl` with
`fifa_state_labels.pkl`. The helper script can evaluate this classifier on the
existing state dataset when project dependencies are installed.

## Re-run Selection Report

From the project root:

```bash
py scripts/evaluate_model_accuracy.py
```

To also evaluate the existing sklearn game-state classifier:

```bash
py scripts/evaluate_model_accuracy.py --state
```

The script does not retrain models or modify datasets. It reads saved detector
metrics from `runs/detect` and, with `--state`, evaluates the existing
game-state classifier against a deterministic 20% holdout split of `dataset/`.
