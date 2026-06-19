import cv2
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = joblib.load(PROJECT_ROOT / "fifa_state_classifier.pkl")
labels = joblib.load(PROJECT_ROOT / "fifa_state_labels.pkl")

video_path = PROJECT_ROOT / "EA SPORTS FIFA 15 2026-06-13 17-11-29.mp4"

ACTIONABLE = {
    "corner",
    "free_kick",
    "goal_kick",
    "kickoff",
    "open_play",
    "penalty",
    "throw_in"
}

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

    can_act = state in ACTIONABLE

    color = (0, 255, 0) if can_act else (0, 0, 255)
    status = "AI ACTIVE" if can_act else "AI WAITING"

    cv2.putText(
        frame,
        f"STATE: {state}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    cv2.putText(
        frame,
        status,
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    cv2.imshow("FIFA State Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
