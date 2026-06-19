# Combined FIFA Object Detector

The combined detector exists to replace three YOLO inference passes with one
YOLO model. It detects gameplay objects that currently require separate player,
ball, and controlled-player-indicator models.

Game-state classification remains separate and is not part of this YOLO model.

## Classes

```text
0: player
1: goalkeeper
2: referee
3: ball
4: controlled_player_indicator
```

## Dataset Merge

The build script reads the existing datasets without modifying them:

- `player_dataset`: keeps classes `0`, `1`, and `2`.
- `ball_dataset`: remaps class `0` to `3`.
- `indicator_dataset`: remaps class `0` to `4`.

Existing `train` and `val` splits are preserved. Images and labels are copied
into `fifa_objects_dataset/`. If a matching label is missing, the image is
skipped with a warning. If the same image filename appears in multiple source
datasets, labels are merged into the same combined label file when possible.

Build the dataset:

```bash
py src/datasets/build_fifa_objects_dataset.py
```

## Config

The YOLO config is:

```text
configs/fifa_objects.yaml
```

It points at:

```text
fifa_objects_dataset/
```

## Train

Train the combined YOLOv8 detector:

```bash
py src/training/train_fifa_objects.py
```

The training constants are near the top of
`src/training/train_fifa_objects.py`:

- `EPOCHS`
- `IMGSZ`
- `BATCH`
- `RUN_NAME`

Training output is expected at:

```text
runs/detect/fifa_objects_v1/
```

## Test

After training, test on a FIFA video:

```bash
py src/inference/test_fifa_objects_video.py
```

The test script draws all five classes and prints per-frame FPS plus inference
time. Compare this against the old pipeline by running the old three-model loop
and comparing FPS/inference time on the same video.

## Old Pipeline

The old separate-model pipeline still exists. This combined model is additive:
it does not delete or overwrite the player, ball, indicator, or game-state
models.
