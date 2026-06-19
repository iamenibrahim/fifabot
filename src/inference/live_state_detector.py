import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "ball_v3" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "EA SPORTS FIFA 15 2026-05-27 18-28-58.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "results" / "videos" / "ball_kalman_output.mp4"

CONFIDENCE = 0.40
IMG_SIZE = 928

model = YOLO(str(MODEL_PATH))

# Kalman state:
# [x, y, vx, vy]
kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)

initialized = False
missed_frames = 0
MAX_MISSED_FRAMES = 15


def get_ball_detections(result):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        detections.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": (cx, cy),
            "conf": conf
        })

    return detections


def choose_detection(detections, predicted_xy):
    if len(detections) == 1:
        return detections[0]

    px, py = predicted_xy

    def distance_to_prediction(det):
        cx, cy = det["center"]
        return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

    return min(detections, key=distance_to_prediction)


cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # Predict next position
    prediction = kalman.predict()
    pred_x = float(prediction[0, 0])
    pred_y = float(prediction[1, 0])

    results = model(
        frame,
        conf=CONFIDENCE,
        imgsz=IMG_SIZE,
        verbose=False
    )

    detections = get_ball_detections(results[0])

    used_detection = None
    status = ""

    if detections:
        if not initialized:
            first = detections[0]
            cx, cy = first["center"]

            kalman.statePost = np.array([
                [cx],
                [cy],
                [0],
                [0]
            ], dtype=np.float32)

            initialized = True
            missed_frames = 0
            used_detection = first
            status = "YOLO INIT"

        else:
            used_detection = choose_detection(detections, (pred_x, pred_y))
            cx, cy = used_detection["center"]

            measurement = np.array([
                [np.float32(cx)],
                [np.float32(cy)]
            ])

            kalman.correct(measurement)
            missed_frames = 0
            status = "YOLO + KALMAN UPDATE"

    else:
        missed_frames += 1

        if initialized and missed_frames <= MAX_MISSED_FRAMES:
            status = "KALMAN PREDICT"
        else:
            status = "NO BALL"

    # Draw YOLO detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            f"YOLO {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            2
        )

    # Draw selected YOLO ball
    if used_detection is not None:
        x1, y1, x2, y2 = used_detection["bbox"]
        cx, cy = used_detection["center"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

    # Draw Kalman prediction
    if initialized and missed_frames <= MAX_MISSED_FRAMES:
        cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (255, 0, 0), 2)
        cv2.putText(
            frame,
            "KALMAN",
            (int(pred_x) + 10, int(pred_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    cv2.putText(
        frame,
        f"Frame: {frame_id} | {status} | detections: {len(detections)} | missed: {missed_frames}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

    cv2.imshow("YOLO + Kalman Ball Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved: {OUTPUT_VIDEO}")
