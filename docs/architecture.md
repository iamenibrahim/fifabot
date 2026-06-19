# Architecture

## Components

- Ball detector: YOLO training and video experiments in `src/training/` and
  `src/detection/`.
- Player detector: YOLO config in `configs/players.yaml`, training in
  `src/training/train_players_yolo.py`, and tracking tests in
  `src/inference/test_players_video.py`.
- Player indicator detector: labeling in
  `src/labeling/label_player_indicator.py` and inference in
  `src/inference/test_indicator_video.py`.
- State classifier: classic ML and CNN/ResNet experiments in
  `src/training/` and `src/inference/`.
- World state: `src/world_state/world_state_detector.py` combines ball,
  player, and indicator detections into video overlays and JSONL state output.

## Data Flow

1. Raw FIFA clips are sampled into frame folders.
2. Labeling scripts create YOLO labels in detector-specific dataset folders.
3. Training scripts consume `configs/*.yaml` and write YOLO runs under `runs/`.
4. Inference scripts load trained weights and produce videos or metrics.
5. The world-state detector combines the three detector streams and writes
   `results/videos/world_state_output.mp4` plus
   `results/metrics/world_state_output.jsonl`.

## Safe Migration Plan

The target layout includes `models/` and `data/`, but existing large assets were
not moved in this cleanup pass. Moving them safely should be a separate change:

1. Inventory every script and config that references a dataset, video, or model
   weight.
2. Copy assets into the new target first, leaving originals in place.
3. Update path constants and YOLO config files to the copied locations.
4. Run a smoke test for each detector and labeler.
5. Compare file counts and label counts between old and new dataset locations.
6. Only after verification, decide whether to remove or archive old locations.

Suggested future homes:

- `data/raw_videos/`: source gameplay recordings.
- `data/frames/`: generated frame dumps.
- `data/datasets/ball/`, `data/datasets/player/`,
  `data/datasets/indicator/`, `data/datasets/state/`: labeled datasets.
- `models/yolo/` and `models/state/`: trained weights/checkpoints.

## Current Large Asset Locations

- Raw videos: project root and `saved_clips_TRAINING/`.
- Datasets and labels: `dataset/`, `ball_dataset/`, `player_dataset/`,
  `indicator_dataset/`, `frames*`.
- Model weights/checkpoints: root `*.pt`, `*.pth`, `*.pkl` and YOLO
  `runs/`.
- Generated outputs: `results/`.

## Path Policy

Moved Python scripts derive `PROJECT_ROOT` from `__file__` and build paths from
there. This lets scripts continue to work when launched from the project root or
directly by path.
