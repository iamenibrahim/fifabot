import math


NEUTRAL_ACTION = {
    "left_stick_x": 0.0,
    "left_stick_y": 0.0,
    "press_a": False,
    "press_b": False,
    "press_x": False,
    "press_y": False,
    "press_rt": False,
}

CLOSE_TO_BALL_PIXELS = 65.0


def neutral_action() -> dict:
    return dict(NEUTRAL_ACTION)


def _clamp_axis(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def decide_action(world_state: dict) -> dict:
    ball_position = world_state.get("ball", {}).get("position")
    controlled_player = world_state.get("controlled_player")

    if ball_position is None or controlled_player is None:
        return neutral_action()

    controlled_center = controlled_player.get("center")
    if controlled_center is None:
        return neutral_action()

    cx, cy = controlled_center
    bx, by = ball_position

    dx = float(bx) - float(cx)
    dy = float(by) - float(cy)
    distance = math.hypot(dx, dy)

    if distance < 1.0:
        action = neutral_action()
        action["press_b"] = True
        return action

    action = neutral_action()
    action["left_stick_x"] = _clamp_axis(dx / distance)
    action["left_stick_y"] = _clamp_axis(dy / distance)

    if distance <= CLOSE_TO_BALL_PIXELS:
        action["press_b"] = True

    return action
