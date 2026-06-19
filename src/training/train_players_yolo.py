from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = YOLO(str(PROJECT_ROOT / "yolov8s.pt"))

model.train(
    data=str(PROJECT_ROOT / "configs" / "players.yaml"),
    epochs=25,
    imgsz=640,
    batch=2,
    name="player_detector_v1"
)
