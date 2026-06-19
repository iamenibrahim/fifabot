from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = str(PROJECT_ROOT / "configs" / "ball.yaml")

experiments = [
    {
        "model": str(PROJECT_ROOT / "yolov8s.pt"),
        "name": "ball_v6_yolov8s_960",
        "imgsz": 960,
        "batch": 4,
        "epochs": 100
    },
    {
        "model": str(PROJECT_ROOT / "yolov8s.pt"),
        "name": "ball_v7_yolov8s_640",
        "imgsz": 640,
        "batch": 8,
        "epochs": 100
    },
    {
        "model": str(PROJECT_ROOT / "yolov8m.pt"),
        "name": "ball_v8_yolov8m_640",
        "imgsz": 640,
        "batch": 4,
        "epochs": 75
    }
]

for exp in experiments:
    print("\n==============================")
    print("Starting:", exp["name"])
    print("==============================\n")

    model = YOLO(exp["model"])

    model.train(
        data=DATA,
        epochs=exp["epochs"],
        imgsz=exp["imgsz"],
        batch=exp["batch"],
        name=exp["name"]
    )

    print("\nFinished:", exp["name"])
