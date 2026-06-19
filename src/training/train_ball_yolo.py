from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))

model.train(
    data=str(PROJECT_ROOT / "configs" / "ball.yaml"),
    epochs=50,
    imgsz=640,
    batch=8,
    name="ball_v2"
)
