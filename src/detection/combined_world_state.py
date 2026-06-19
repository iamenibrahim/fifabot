import cv2
import math
import sys
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.model_paths import BALL_MODEL_PATH, PLAYER_MODEL_PATH

VIDEO_PATH = PROJECT_ROOT / "EA SPORTS FIFA 15 2026-05-27 18-28-58.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "results" / "videos" / "combined_ball_players_output.mp4"

BALL_CONF = 0.25
PLAYER_CONF = 0.25
IMG_SIZE = 480

ball_model = YOLO(str(BALL_MODEL_PATH))
player_model = YOLO(str(PLAYER_MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ball_results = ball_model(frame, conf=BALL_CONF, imgsz=IMG_SIZE, verbose=False)
    player_results = player_model(frame, conf=PLAYER_CONF, imgsz=IMG_SIZE, verbose=False)

    ball_box = None
    players = []

    # Get best ball only
    if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
        best_ball = max(ball_results[0].boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best_ball.xyxy[0].cpu().numpy()
        ball_box = (int(x1), int(y1), int(x2), int(y2))

    # Get players only, class 0
    if player_results[0].boxes is not None:
        for box in player_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            players.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "conf": conf
            })

    nearest_player = None
    nearest_distance = None

    if ball_box is not None and players:
        bx, by = get_center(ball_box)

        for player in players:
            px, py = get_center(player["box"])
            distance = math.sqrt((px - bx) ** 2 + (py - by) ** 2)

            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_player = player

    # Draw players
    for player in players:
        x1, y1, x2, y2 = player["box"]

        color = (255, 0, 0)

        if player is nearest_player:
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw ball
    if ball_box is not None:
        x1, y1, x2, y2 = ball_box
        bx, by = get_center(ball_box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (bx, by), 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            "BALL",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw nearest player line
    if ball_box is not None and nearest_player is not None:
        bx, by = get_center(ball_box)
        px, py = get_center(nearest_player["box"])

        cv2.line(frame, (bx, by), (px, py), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Nearest player: {nearest_distance:.1f}px",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    out.write(frame)
    cv2.imshow("Combined Ball + Players", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved: {OUTPUT_VIDEO}")
