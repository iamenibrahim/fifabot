import cv2
import json
import math
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.model_paths import BALL_MODEL_PATH, INDICATOR_MODEL_PATH, PLAYER_MODEL_PATH


# =========================
# PATHS — CHANGE THESE
# =========================

VIDEO_PATH = PROJECT_ROOT / "EA SPORTS FIFA 15 2026-05-27 18-28-58.mp4"

OUTPUT_VIDEO = PROJECT_ROOT / "results" / "videos" / "world_state_output.mp4"
OUTPUT_JSONL = PROJECT_ROOT / "results" / "metrics" / "world_state_output.jsonl"


# =========================
# SETTINGS
# =========================

BALL_CONF = 0.40
PLAYER_CONF = 0.25
INDICATOR_CONF = 0.25

# The ball is tiny in FIFA footage. The standalone ball tracker performed
# better at this larger inference size than the old live-loop value of 320.
BALL_IMGSZ = 928
PLAYER_IMGSZ = 320
INDICATOR_IMGSZ = 416

SHOW_VIDEO = True
SAVE_VIDEO = True
SAVE_JSONL = True

MAX_BALL_MISSED_FRAMES = 15


# =========================
# LOAD MODELS
# =========================

ball_model = YOLO(str(BALL_MODEL_PATH))
player_model = YOLO(str(PLAYER_MODEL_PATH))
indicator_model = YOLO(str(INDICATOR_MODEL_PATH))


# =========================
# UTILITY FUNCTIONS
# =========================

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def bbox_top_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y1)


def distance(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def run_detector(model, frame, conf, imgsz):
    """
    Runs a YOLO model and returns detections as dictionaries.
    """
    results = model(
        frame,
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )

    detections = []

    boxes = results[0].boxes

    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        bbox = (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

        detections.append({
            "class_id": cls_id,
            "confidence": score,
            "bbox": bbox,
            "center": bbox_center(bbox),
            "bottom_center": bbox_bottom_center(bbox),
            "top_center": bbox_top_center(bbox)
        })

    return detections


def choose_best_detection(detections):
    if not detections:
        return None

    return max(detections, key=lambda d: d["confidence"])


def choose_ball_detection(ball_detections, predicted_ball=None):
    """
    If there is one ball, use it.
    If multiple balls, choose the one closest to Kalman prediction.
    If no prediction exists, choose highest confidence.
    """
    if not ball_detections:
        return None

    if len(ball_detections) == 1:
        return ball_detections[0]

    if predicted_ball is not None:
        return min(
            ball_detections,
            key=lambda d: distance(d["center"], predicted_ball)
        )

    return choose_best_detection(ball_detections)

def find_controlled_player(indicator_detection, player_detections):
    """
    Pair the player indicator with the correct player box.

    Better logic:
    - Indicator should be above the player.
    - Player top-center should be close horizontally to indicator center.
    - Player top should be below the indicator.
    - Penalize huge boxes that accidentally cover multiple players.
    """

    if indicator_detection is None or not player_detections:
        return None

    ix, iy = indicator_detection["center"]
    ix1, iy1, ix2, iy2 = indicator_detection["bbox"]
    indicator_bottom_y = iy2

    best_player = None
    best_score = float("inf")

    for player in player_detections:
        x1, y1, x2, y2 = player["bbox"]

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h

        player_top_x = (x1 + x2) / 2
        player_top_y = y1

        horizontal_dist = abs(player_top_x - ix)
        vertical_gap = player_top_y - indicator_bottom_y

        # Must be below indicator
        if vertical_gap < -20:
            continue

        # Too far below indicator
        if vertical_gap > 220:
            continue

        # Too far sideways
        if horizontal_dist > 120:
            continue

        # Penalize giant boxes that likely cover multiple players
        size_penalty = 0
        if box_h > 220:
            size_penalty += 250
        if box_w > 130:
            size_penalty += 200
        if box_area > 25000:
            size_penalty += 250

        # Prefer close horizontal alignment and reasonable vertical gap
        score = (
            horizontal_dist * 2.0 +
            vertical_gap * 0.8 +
            size_penalty
        )

        if score < best_score:
            best_score = score
            best_player = player

    return best_player

def find_nearest_player_to_ball(ball_position, player_detections):
    """
    Uses player bottom-center because that represents the feet better than box center.
    """
    if ball_position is None or not player_detections:
        return None

    return min(
        player_detections,
        key=lambda p: distance(ball_position, p["bottom_center"])
    )


def clean_detection_for_json(det):
    if det is None:
        return None

    return {
        "class_id": int(det["class_id"]),
        "confidence": float(det["confidence"]),
        "bbox": [int(v) for v in det["bbox"]],
        "center": [float(det["center"][0]), float(det["center"][1])],
        "bottom_center": [float(det["bottom_center"][0]), float(det["bottom_center"][1])],
        "top_center": [float(det["top_center"][0]), float(det["top_center"][1])]
    }


# =========================
# KALMAN FILTER FOR BALL
# =========================

class BallKalman:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)

        # State: x, y, vx, vy
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.missed_frames = 0

    def predict(self):
        if not self.initialized:
            return None

        prediction = self.kalman.predict()
        x = float(prediction[0, 0])
        y = float(prediction[1, 0])

        return (x, y)

    def update(self, measured_xy):
        x, y = measured_xy

        if not self.initialized:
            self.kalman.statePost = np.array([
                [np.float32(x)],
                [np.float32(y)],
                [0],
                [0]
            ], dtype=np.float32)

            self.initialized = True
            self.missed_frames = 0
            return

        measurement = np.array([
            [np.float32(x)],
            [np.float32(y)]
        ])

        self.kalman.correct(measurement)
        self.missed_frames = 0

    def miss(self):
        self.missed_frames += 1

    def has_valid_prediction(self):
        return self.initialized and self.missed_frames <= MAX_BALL_MISSED_FRAMES


# =========================
# DRAWING FUNCTIONS
# =========================

def draw_box(frame, det, label, color, thickness=2):
    if det is None:
        return

    x1, y1, x2, y2 = det["bbox"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


def draw_point(frame, point, color, radius=6, label=None):
    if point is None:
        return

    x, y = int(point[0]), int(point[1])

    cv2.circle(frame, (x, y), radius, color, -1)

    if label:
        cv2.putText(
            frame,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


# =========================
# MAIN LOOP
# =========================

def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None

    if SAVE_VIDEO:
        out = cv2.VideoWriter(
            OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

    jsonl_file = None

    if SAVE_JSONL:
        jsonl_file = open(OUTPUT_JSONL, "w", encoding="utf-8")

    ball_tracker = BallKalman()

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1
        timestamp = frame_id / fps if fps else 0

        # -------------------------
        # 1. Run all three models
        # -------------------------

        ball_detections = run_detector(
            ball_model,
            frame,
            conf=BALL_CONF,
            imgsz=BALL_IMGSZ
        )

        player_detections = run_detector(
            player_model,
            frame,
            conf=PLAYER_CONF,
            imgsz=PLAYER_IMGSZ
        )

        indicator_detections = run_detector(
            indicator_model,
            frame,
            conf=INDICATOR_CONF,
            imgsz=INDICATOR_IMGSZ
        )

        # -------------------------
        # 2. Ball selection + Kalman
        # -------------------------

        predicted_ball = ball_tracker.predict()

        selected_ball = choose_ball_detection(
            ball_detections,
            predicted_ball=predicted_ball
        )

        ball_position = None
        ball_source = "none"

        if selected_ball is not None:
            ball_position = selected_ball["center"]
            ball_tracker.update(ball_position)
            ball_source = "yolo"
        else:
            ball_tracker.miss()

            if ball_tracker.has_valid_prediction():
                ball_position = predicted_ball
                ball_source = "kalman"
            else:
                ball_position = None
                ball_source = "none"

        # -------------------------
        # 3. Indicator selection
        # -------------------------

        selected_indicator = choose_best_detection(indicator_detections)

        # -------------------------
        # 4. Controlled player pairing
        # -------------------------

        controlled_player = find_controlled_player(
            selected_indicator,
            player_detections
        )

        # -------------------------
        # 5. Nearest player to ball
        # -------------------------

        nearest_player_to_ball = find_nearest_player_to_ball(
            ball_position,
            player_detections
        )

        # -------------------------
        # 6. Build world state
        # -------------------------

        controlled_player_center = (
            controlled_player["center"]
            if controlled_player is not None
            else None
        )

        distance_controlled_to_ball = (
            distance(controlled_player_center, ball_position)
            if controlled_player_center is not None and ball_position is not None
            else None
        )

        world_state = {
            "frame": frame_id,
            "timestamp": timestamp,

            "ball": {
                "position": [float(ball_position[0]), float(ball_position[1])] if ball_position else None,
                "source": ball_source,
                "detection": clean_detection_for_json(selected_ball)
            },

            "indicator": clean_detection_for_json(selected_indicator),

            "controlled_player": clean_detection_for_json(controlled_player),

            "nearest_player_to_ball": clean_detection_for_json(nearest_player_to_ball),

            "distance_controlled_to_ball": float(distance_controlled_to_ball)
            if distance_controlled_to_ball is not None
            else None,

            "players": [
                clean_detection_for_json(p)
                for p in player_detections
            ],

            "counts": {
                "ball_detections": len(ball_detections),
                "player_detections": len(player_detections),
                "indicator_detections": len(indicator_detections)
            }
        }

        if jsonl_file:
            jsonl_file.write(json.dumps(world_state) + "\n")

        # -------------------------
        # 7. Draw overlay
        # -------------------------

        # Draw all players
        for p in player_detections:
            label = f"player {p['confidence']:.2f}"
            draw_box(frame, p, label, color=(255, 80, 0), thickness=2)

        # Highlight nearest player to ball
        if nearest_player_to_ball is not None:
            draw_box(
                frame,
                nearest_player_to_ball,
                "nearest_to_ball",
                color=(0, 255, 255),
                thickness=3
            )

        # Highlight controlled player
        if controlled_player is not None:
            draw_box(
                frame,
                controlled_player,
                "CONTROLLED",
                color=(255, 0, 255),
                thickness=4
            )

        # Draw indicator
        if selected_indicator is not None:
            draw_box(
                frame,
                selected_indicator,
                f"indicator {selected_indicator['confidence']:.2f}",
                color=(0, 165, 255),
                thickness=3
            )

        # Draw ball detections
        for b in ball_detections:
            draw_box(
                frame,
                b,
                f"ball {b['confidence']:.2f}",
                color=(0, 255, 0),
                thickness=2
            )

        # Draw final ball position
        if ball_position is not None:
            if ball_source == "yolo":
                draw_point(frame, ball_position, color=(0, 255, 0), radius=7, label="BALL")
            elif ball_source == "kalman":
                draw_point(frame, ball_position, color=(255, 255, 0), radius=7, label="BALL_PRED")

        # Draw status box
        status_text = (
            f"Frame {frame_id} | "
            f"Players: {len(player_detections)} | "
            f"Ball: {ball_source} | "
            f"Indicators: {len(indicator_detections)}"
        )

        cv2.rectangle(frame, (20, 20), (900, 65), (0, 0, 0), -1)

        cv2.putText(
            frame,
            status_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if distance_controlled_to_ball is not None:
            cv2.putText(
                frame,
                f"Controlled-to-ball distance: {distance_controlled_to_ball:.1f}px",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

        if SAVE_VIDEO:
            out.write(frame)

        if SHOW_VIDEO:
            cv2.imshow("FIFA World State Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    if out:
        out.release()

    if jsonl_file:
        jsonl_file.close()

    cv2.destroyAllWindows()

    print("Done.")
    if SAVE_VIDEO:
        print(f"Saved video: {OUTPUT_VIDEO}")
    if SAVE_JSONL:
        print(f"Saved world state: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
