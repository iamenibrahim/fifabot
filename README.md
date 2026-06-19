# FIFA Bot Computer Vision

This project contains experiments and scripts for FIFA gameplay computer vision:
ball detection, player detection, controlled-player indicator detection, state
classification, and world-state generation.

## Project Layout

```text
fifabot/
  src/
    detection/      # ball/player detector experiments and helper functions
    labeling/       # manual labelers and dataset split scripts
    training/       # YOLO and state-classifier training scripts
    inference/      # video/live prediction scripts
    world_state/    # combined world-state detector
  configs/          # YOLO dataset and tracker configs
  docs/             # architecture and migration notes
  results/
    screenshots/    # small visual outputs
    metrics/        # CSV/JSONL outputs
    videos/         # generated videos, ignored by git
  scripts/          # utility scripts
  models/           # future model-weight home, ignored by git
  data/             # future dataset/video home, ignored by git
```

Large assets currently remain in their original locations to avoid changing
dataset labels, weights, or video references unexpectedly. See
`docs/architecture.md` for the safe migration plan.

## Common Commands

Run scripts from the project root:

```powershell
python src\world_state\world_state_detector.py
python src\training\train_ball_yolo.py
python src\labeling\label_ball.py
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Notes

- `models/`, `data/`, raw videos, datasets, weights, and YOLO `runs/` are
  ignored by git.
- Existing raw videos, datasets, labels, and model weights were not modified.
- Path references in moved scripts are anchored to the project root.
