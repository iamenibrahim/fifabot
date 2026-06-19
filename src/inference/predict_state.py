import cv2
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = joblib.load(PROJECT_ROOT / "fifa_state_classifier.pkl")
labels = joblib.load(PROJECT_ROOT / "fifa_state_labels.pkl")

video_path = PROJECT_ROOT / "fifa_clip.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    prediction = model.predict([gray.flatten()])[0]
    state = labels[prediction]

    cv2.putText(
        frame,
        f"STATE: {state}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("FIFA State Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
