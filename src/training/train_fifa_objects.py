from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = PROJECT_ROOT / "yolov8s.pt"
DATA = PROJECT_ROOT / "configs" / "fifa_objects.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 8
RUN_NAME = "fifa_objects_v1"


def main():
    model = YOLO(str(MODEL))
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=RUN_NAME,
    )


if __name__ == "__main__":
    main()
