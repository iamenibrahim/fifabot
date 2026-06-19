from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chosen from existing saved YOLO validation metrics in runs/detect/*/results.csv.
BALL_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "ball_v3" / "weights" / "best.pt"
PLAYER_MODEL_PATH = (
    PROJECT_ROOT / "runs" / "detect" / "player_detector_v1-5" / "weights" / "best.pt"
)
INDICATOR_MODEL_PATH = (
    PROJECT_ROOT / "runs" / "detect" / "indicator_v1" / "weights" / "best.pt"
)

# Existing trained game-state classifier. Use scripts/evaluate_model_accuracy.py
# in an environment with project dependencies installed to compare state models.
STATE_CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "fifa_state_classifier.pkl"
STATE_CLASSIFIER_LABELS_PATH = PROJECT_ROOT / "fifa_state_labels.pkl"
