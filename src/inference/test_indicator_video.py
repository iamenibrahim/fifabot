from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "indicator_v1" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "fifa_clip4.mp4"

model = YOLO(str(MODEL_PATH))

model.predict(
    source=str(VIDEO_PATH),
    conf=0.35,
    save=True,
    show=True,
    imgsz=416,
    line_width=1
)
