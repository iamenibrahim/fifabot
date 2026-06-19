import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load model
model = YOLO(str(PROJECT_ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt"))

# Setup Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

# Settings
MAX_DISPLACEMENT = 80  # max pixels ball can move per frame
MIN_CONF = 0.25

cap = cv2.VideoCapture(str(PROJECT_ROOT / "fifa_clip2.mp4"))

last_pos = None
kalman_initialized = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=MIN_CONF, verbose=False)

    best_box = None
    best_conf = 0

    for box in results[0].boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Reject if ball teleported
        if last_pos is not None:
            dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
            if dist > MAX_DISPLACEMENT:
                continue

        if conf > best_conf:
            best_conf = conf
            best_box = (x1, y1, x2, y2, cx, cy)

    if best_box is not None:
        x1, y1, x2, y2, cx, cy = best_box

        # Feed into Kalman
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not kalman_initialized:
            kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            kalman_initialized = True
        kalman.correct(measurement)
        predicted = kalman.predict()

        last_pos = (cx, cy)

        # Draw detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ball {best_conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif kalman_initialized:
        # Ball not detected — use Kalman prediction
        predicted = kalman.predict()
        px, py = int(predicted[0][0]), int(predicted[1][0])
        cv2.circle(frame, (px, py), 6, (0, 165, 255), -1)  # orange = predicted
        cv2.putText(frame, "predicted", (px + 8, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    cv2.imshow("Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
