from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ball_model = YOLO(str(PROJECT_ROOT / "runs" / "detect" / "ball_v3" / "weights" / "best_openvino_model"))
results = model(frame, device="GPU", verbose=False)
