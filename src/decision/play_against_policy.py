import math

from decision.simple_policy import neutral_action


ACTIONABLE_STATES = {
    "corner",
    "free_kick",
    "goal_kick",
    "kickoff",
    "open_play",
    "penalty",
    "throw_in",
}

CHASE_DISTANCE_PIXELS = 180.0
TACKLE_DISTANCE_PIXELS = 55.0


def _clamp_axis(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _direction(from_point, to_point):
    fx, fy = from_point
    tx, ty = to_point
    dx = float(tx) - float(fx)
    dy = float(ty) - float(fy)
    distance = math.hypot(dx, dy)
    if distance < 1.0:
        return 0.0, 0.0, distance
    return _clamp_axis(dx / distance), _clamp_axis(dy / distance), distance


def decide_play_against_action(world_state: dict) -> dict:
    action = neutral_action()

    game_state = world_state.get("game_state", {})
    state_name = game_state.get("state")
    if state_name is not None and state_name not in ACTIONABLE_STATES:
        return action

    ball_position = world_state.get("ball", {}).get("position")
    controlled_player = world_state.get("controlled_player")
    if ball_position is None or controlled_player is None:
        return action

    controlled_center = controlled_player.get("center")
    if controlled_center is None:
        return action

    move_x, move_y, distance = _direction(controlled_center, ball_position)
    action["left_stick_x"] = move_x
    action["left_stick_y"] = move_y

    # Hold sprint while chasing, then tackle/clear near the ball.
    action["press_rt"] = distance > CHASE_DISTANCE_PIXELS
    action["press_b"] = distance <= TACKLE_DISTANCE_PIXELS

    return action
