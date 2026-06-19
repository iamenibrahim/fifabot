# Control Loop

The first live bot loop connects the existing vision stack to a virtual Xbox
controller:

```text
frame capture -> world_state -> simple_policy -> vgamepad output
```

The current behavior is intentionally simple. The loop uses the existing trained
ball, player, controlled-player indicator, and game-state models. If the bot can
see both the ball and the controlled player, it moves the left stick in the
direction from the controlled player toward the ball. If either target is
missing, it sends neutral input. When the controlled player is close to the
ball, the policy presses `B` as a basic placeholder action.

## Files

- `src/decision/simple_policy.py`: rule-based decision policy.
- `src/control/controller_output.py`: virtual Xbox controller adapter.
- `src/main_live_bot.py`: live control loop with debug overlay.
- `src/world_state/state_classifier.py`: wrapper around the existing trained
  game-state classifier files.

## Existing Models Used

- Ball detector: `runs/detect/ball_v3/weights/best.pt`
- Player detector: `runs/detect/player_detector_v1-6/weights/best.pt`
- Controlled-player indicator detector:
  `runs/detect/indicator_v1/weights/best.pt`
- Game-state classifier: `fifa_state_classifier.pkl` and
  `fifa_state_labels.pkl`

## Install Dependencies

Install the Python dependencies:

```bash
pip install -r requirements.txt
pip install vgamepad
```

On Windows, `vgamepad` usually requires ViGEmBus. If controller creation fails
or no virtual controller appears, install ViGEmBus and restart the machine.

## Run

From the project root:

```bash
py src/main_live_bot.py
```

If `py` is not available on your machine, use whichever Python executable has
the dependencies installed.

## Stop Safely

Press `q` in the debug window to quit. The loop uses `try/finally`, so it resets
the virtual controller to neutral on exit. If the process is interrupted from
PowerShell, the same cleanup path should still run for normal `Ctrl+C`
interrupts.

## Notes

- The live bot does not retrain or modify any model.
- The current frame source is the same video path used by the world-state
  detector. Swap the `cv2.VideoCapture(...)` source in `src/main_live_bot.py`
  when you are ready to capture from a live device.
- This is a rule-based starter loop, not reinforcement learning or tactics.
