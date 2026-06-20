# Play-Against Bot

This mode runs a simple CPU/GPU vision loop that controls a virtual Xbox
controller so you can play against the bot in FIFA.

It is intentionally rule-based:

- capture the screen
- detect players, ball, and controlled-player indicator
- classify the game state separately
- move the selected bot player toward the ball
- hold sprint while chasing
- press `B` when close enough to tackle/clear

It does not use reinforcement learning and it does not replace the existing
training or three-model pipeline.

## Requirements

Install Python dependencies:

```powershell
py -m pip install -r requirements.txt
py -m pip install vgamepad
```

On Windows, `vgamepad` usually requires ViGEmBus.

## Model Choice

The play-against bot prefers the combined detector:

```text
runs/detect/fifa_objects_v1/weights/best.pt
best.pt
```

If neither exists, it falls back to the old separate player, ball, and indicator
detectors.

The game-state classifier remains separate.

## Run

Start FIFA first and make sure the bot side is controlled by the virtual Xbox
controller. Then run:

```powershell
py src/main_play_against_bot.py
```

Controls:

- `p`: pause/resume controller output
- `q`: quit and reset the controller

The bot starts paused so you can switch back into FIFA safely. Press `p` in the
debug window when you are ready.

## Debug Overlay

The overlay shows:

- loop FPS
- game state
- stick direction
- `B` and `RT` button state
- player, indicator, and controlled-player detection counts

If `ind=0`, the indicator is not being detected. If `ctrl=0`, the bot does not
currently know which player it controls.

## Safety

The controller is reset in a `finally` block on exit. If anything feels wrong,
click the debug window and press `q`, or press `p` to pause output.
