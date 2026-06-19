from ultralytics import YOLO
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = YOLO(str(PROJECT_ROOT / "runs" / "detect" / "ball_v3" / "weights" / "best.pt"))

img = cv2.imread(str(PROJECT_ROOT / "results" / "screenshots" / "frame_0462.jpg"))
results = model(img, conf=0.3)

for box in results[0].boxes:
    print(f"conf: {float(box.conf):.3f}")

cv2.imshow("test", results[0].plot())
cv2.waitKey(0)
