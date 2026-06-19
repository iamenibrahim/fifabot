import cv2

from control.controller_output import XboxController
from decision.simple_policy import decide_action, neutral_action
from world_state.state_classifier import GameStateClassifier
from world_state import world_state_detector as detector


WINDOW_NAME = "FIFA Bot Control Loop"


def build_world_state(
    frame,
    ball_tracker,
    state_classifier: GameStateClassifier,
    frame_id: int,
    fps: float,
) -> dict:
    game_state = state_classifier.predict(frame)

    ball_detections = detector.run_detector(
        detector.ball_model,
        frame,
        conf=detector.BALL_CONF,
        imgsz=detector.BALL_IMGSZ,
    )

    player_detections = detector.run_detector(
        detector.player_model,
        frame,
        conf=detector.PLAYER_CONF,
        imgsz=detector.PLAYER_IMGSZ,
    )

    indicator_detections = detector.run_detector(
        detector.indicator_model,
        frame,
        conf=detector.INDICATOR_CONF,
        imgsz=detector.INDICATOR_IMGSZ,
    )

    predicted_ball = ball_tracker.predict()
    selected_ball = detector.choose_ball_detection(
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

    selected_indicator = detector.choose_best_detection(indicator_detections)
    controlled_player = detector.find_controlled_player(
        selected_indicator,
        player_detections,
    )
    nearest_player_to_ball = detector.find_nearest_player_to_ball(
        ball_position,
        player_detections,
    )

    controlled_player_center = (
        controlled_player["center"] if controlled_player is not None else None
    )
    controlled_distance = (
        detector.distance(controlled_player_center, ball_position)
        if controlled_player_center is not None and ball_position is not None
        else None
    )

    return {
        "frame": frame_id,
        "timestamp": frame_id / fps if fps else 0.0,
        "game_state": game_state,
        "ball": {
            "position": [float(ball_position[0]), float(ball_position[1])]
            if ball_position
            else None,
            "source": ball_source,
            "detection": detector.clean_detection_for_json(selected_ball),
        },
        "indicator": detector.clean_detection_for_json(selected_indicator),
        "controlled_player": detector.clean_detection_for_json(controlled_player),
        "nearest_player_to_ball": detector.clean_detection_for_json(
            nearest_player_to_ball
        ),
        "distance_controlled_to_ball": float(controlled_distance)
        if controlled_distance is not None
        else None,
        "players": [
            detector.clean_detection_for_json(player) for player in player_detections
        ],
        "counts": {
            "ball_detections": len(ball_detections),
            "player_detections": len(player_detections),
            "indicator_detections": len(indicator_detections),
        },
    }


def draw_debug_overlay(frame, world_state: dict, action: dict):
    ball_position = world_state["ball"]["position"]
    controlled_player = world_state["controlled_player"]
    game_state = world_state.get("game_state", {})

    if ball_position is not None:
        detector.draw_point(frame, ball_position, color=(0, 255, 0), label="BALL")

    if controlled_player is not None:
        detector.draw_box(frame, controlled_player, "CONTROLLED", (255, 0, 255), 3)
        if ball_position is not None:
            cx, cy = controlled_player["center"]
            bx, by = ball_position
            cv2.line(frame, (int(cx), int(cy)), (int(bx), int(by)), (0, 255, 255), 2)

    status = (
        f"stick=({action['left_stick_x']:.2f}, {action['left_stick_y']:.2f}) "
        f"B={int(action['press_b'])} "
        f"ball={world_state['ball']['source']} "
        f"state={game_state.get('state', 'unknown')}"
    )

    cv2.rectangle(frame, (20, 20), (760, 68), (0, 0, 0), -1)
    cv2.putText(
        frame,
        status,
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )


def main():
    controller = XboxController()
    cap = None

    try:
        cap = cv2.VideoCapture(str(detector.VIDEO_PATH))
        ball_tracker = detector.BallKalman()
        state_classifier = GameStateClassifier()

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {detector.VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            world_state = build_world_state(
                frame,
                ball_tracker,
                state_classifier,
                frame_id,
                fps,
            )
            action = decide_action(world_state)
            controller.apply_action(action)
            draw_debug_overlay(frame, world_state, action)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        controller.apply_action(neutral_action())
        controller.reset()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
