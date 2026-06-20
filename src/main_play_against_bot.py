import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control.controller_output import XboxController
from decision.play_against_policy import decide_play_against_action
from decision.simple_policy import neutral_action
from world_state.state_classifier import GameStateClassifier
from world_state import world_state_detector as separate_detector


WINDOW_NAME = "FIFA Play-Against Bot"
COMBINED_MODEL_CANDIDATES = [
    PROJECT_ROOT / "runs" / "detect" / "fifa_objects_v1" / "weights" / "best.pt",
    PROJECT_ROOT / "best.pt",
]

COMBINED_CONF = 0.25
COMBINED_IMGSZ = 640
MAX_CONTROLLED_PLAYER_MISSES = 12


class ControlledPlayerMemory:
    def __init__(self):
        self.last_center = None
        self.missed_frames = 0

    def update(self, controlled_player, player_detections):
        if controlled_player is not None:
            self.last_center = controlled_player["center"]
            self.missed_frames = 0
            return controlled_player

        if self.last_center is None or not player_detections:
            return None

        self.missed_frames += 1
        if self.missed_frames > MAX_CONTROLLED_PLAYER_MISSES:
            self.last_center = None
            return None

        fallback = min(
            player_detections,
            key=lambda player: separate_detector.distance(
                self.last_center,
                player["center"],
            ),
        )
        self.last_center = fallback["center"]
        return fallback


def load_combined_model():
    model_path = next((path for path in COMBINED_MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        return None
    print(f"Using combined detector: {model_path}")
    return YOLO(str(model_path))


def capture_frame(sct, monitor):
    screenshot = np.array(sct.grab(monitor))
    return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)


def detection_from_box(box):
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    bbox = (int(x1), int(y1), int(x2), int(y2))
    return {
        "class_id": class_id,
        "confidence": confidence,
        "bbox": bbox,
        "center": separate_detector.bbox_center(bbox),
        "bottom_center": separate_detector.bbox_bottom_center(bbox),
        "top_center": separate_detector.bbox_top_center(bbox),
    }


def run_combined_detector(model, frame):
    results = model(frame, conf=COMBINED_CONF, imgsz=COMBINED_IMGSZ, verbose=False)
    players = []
    balls = []
    indicators = []

    boxes = results[0].boxes
    if boxes is None:
        return players, balls, indicators

    for box in boxes:
        detection = detection_from_box(box)
        class_id = detection["class_id"]
        if class_id in {0, 1, 2}:
            players.append(detection)
        elif class_id == 3:
            balls.append(detection)
        elif class_id == 4:
            indicators.append(detection)

    return players, balls, indicators


def run_separate_detectors(frame):
    balls = separate_detector.run_detector(
        separate_detector.ball_model,
        frame,
        separate_detector.BALL_CONF,
        separate_detector.BALL_IMGSZ,
    )
    players = separate_detector.run_detector(
        separate_detector.player_model,
        frame,
        separate_detector.PLAYER_CONF,
        separate_detector.PLAYER_IMGSZ,
    )
    indicators = separate_detector.run_detector(
        separate_detector.indicator_model,
        frame,
        separate_detector.INDICATOR_CONF,
        separate_detector.INDICATOR_IMGSZ,
    )
    return players, balls, indicators


def build_world_state(
    frame,
    detector_model,
    ball_tracker,
    controlled_memory,
    state_classifier,
    frame_id,
    fps,
):
    if detector_model is not None:
        player_detections, ball_detections, indicator_detections = run_combined_detector(
            detector_model,
            frame,
        )
    else:
        player_detections, ball_detections, indicator_detections = run_separate_detectors(
            frame
        )

    predicted_ball = ball_tracker.predict()
    selected_ball = separate_detector.choose_ball_detection(
        ball_detections,
        predicted_ball=predicted_ball,
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

    selected_indicator = separate_detector.choose_best_detection(indicator_detections)
    controlled_player = separate_detector.find_controlled_player(
        selected_indicator,
        player_detections,
    )
    controlled_player = controlled_memory.update(controlled_player, player_detections)

    return {
        "frame": frame_id,
        "timestamp": frame_id / fps if fps else 0.0,
        "game_state": state_classifier.predict(frame),
        "ball": {
            "position": [float(ball_position[0]), float(ball_position[1])]
            if ball_position
            else None,
            "source": ball_source,
            "detection": separate_detector.clean_detection_for_json(selected_ball),
        },
        "indicator": separate_detector.clean_detection_for_json(selected_indicator),
        "controlled_player": separate_detector.clean_detection_for_json(
            controlled_player
        ),
        "players": [
            separate_detector.clean_detection_for_json(player)
            for player in player_detections
        ],
        "counts": {
            "ball_detections": len(ball_detections),
            "player_detections": len(player_detections),
            "indicator_detections": len(indicator_detections),
        },
    }


def draw_debug(frame, world_state, action, paused, loop_fps):
    ball_position = world_state["ball"]["position"]
    controlled_player = world_state["controlled_player"]
    indicator = world_state["indicator"]

    if ball_position is not None:
        separate_detector.draw_point(frame, ball_position, (0, 255, 0), label="BALL")
    if indicator is not None:
        separate_detector.draw_box(frame, indicator, "IND", (0, 165, 255), 2)
    if controlled_player is not None:
        separate_detector.draw_box(frame, controlled_player, "BOT", (255, 0, 255), 3)
        if ball_position is not None:
            cx, cy = controlled_player["center"]
            bx, by = ball_position
            cv2.line(frame, (int(cx), int(cy)), (int(bx), int(by)), (0, 255, 255), 2)

    status = (
        f"{'PAUSED' if paused else 'RUNNING'} | "
        f"fps={loop_fps:.1f} | "
        f"state={world_state['game_state'].get('state', 'unknown')} | "
        f"stick=({action['left_stick_x']:.2f},{action['left_stick_y']:.2f}) | "
        f"B={int(action['press_b'])} RT={int(action['press_rt'])} | "
        f"players={world_state['counts']['player_detections']} "
        f"ind={world_state['counts']['indicator_detections']} "
        f"ctrl={int(controlled_player is not None)}"
    )
    cv2.rectangle(frame, (20, 20), (1260, 68), (0, 0, 0), -1)
    cv2.putText(
        frame,
        status,
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def main():
    detector_model = load_combined_model()
    if detector_model is None:
        print("Combined detector not found. Falling back to separate detectors.")

    controller = XboxController()
    state_classifier = GameStateClassifier()
    ball_tracker = separate_detector.BallKalman()
    controlled_memory = ControlledPlayerMemory()
    paused = True
    frame_id = 0
    previous_time = time.perf_counter()

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        try:
            while True:
                frame = capture_frame(sct, monitor)
                frame_id += 1

                now = time.perf_counter()
                dt = now - previous_time
                previous_time = now
                loop_fps = 1.0 / dt if dt > 0 else 0.0

                world_state = build_world_state(
                    frame,
                    detector_model,
                    ball_tracker,
                    controlled_memory,
                    state_classifier,
                    frame_id,
                    loop_fps,
                )

                action = neutral_action() if paused else decide_play_against_action(world_state)
                controller.apply_action(action)
                draw_debug(frame, world_state, action, paused, loop_fps)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                    if paused:
                        controller.reset()
        finally:
            controller.apply_action(neutral_action())
            controller.reset()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
