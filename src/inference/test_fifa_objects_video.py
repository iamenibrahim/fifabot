import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "fifa_objects_v1" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "EA SPORTS FIFA 15 2026-05-27 18-28-58.mp4"
CONFIDENCE = 0.25
IMGSZ = 640

CLASS_COLORS = {
    0: (255, 80, 0),
    1: (0, 255, 255),
    2: (255, 255, 255),
    3: (0, 255, 0),
    4: (0, 165, 255),
}


def draw_detection(frame, box, names):
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
    color = CLASS_COLORS.get(class_id, (255, 0, 255))
    label = f"{names[class_id]} {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Combined model not found: {MODEL_PATH}. "
            "Train it with src/training/train_fifa_objects.py first."
        )

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_count = 0
    total_inference_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start = time.perf_counter()
        results = model(frame, conf=CONFIDENCE, imgsz=IMGSZ, verbose=False)
        inference_time = time.perf_counter() - start
        total_inference_time += inference_time

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                draw_detection(frame, box, model.names)

        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0.0
        status = f"FPS {fps:.1f} | avg {avg_fps:.1f} | inference {inference_time * 1000:.1f} ms"
        cv2.rectangle(frame, (20, 20), (760, 64), (0, 0, 0), -1)
        cv2.putText(
            frame,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        print(status)
        cv2.imshow("FIFA Objects Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
