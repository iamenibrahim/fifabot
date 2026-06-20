# Controller Troubleshooting

Before judging the bot AI, verify that FIFA receives virtual Xbox controller
input at all.

## 1. Run The Manual Input Test

Open FIFA in a menu, arena, or match. Then run:

```powershell
py src/control/test_controller_input.py
```

The script sends:

- left stick right for 2 seconds
- left stick up plus `RT` for 2 seconds
- `B` for 0.5 seconds

If FIFA does not react, the problem is controller setup, not computer vision or
policy.

## 2. Check ViGEmBus

`vgamepad` needs ViGEmBus on Windows. If controller creation fails or FIFA never
sees the virtual controller, install/reinstall ViGEmBus and restart Windows.

## 3. Check FIFA Side Assignment

FIFA must assign one side to the virtual Xbox controller. If your keyboard or
physical controller owns both sides, the bot can send input forever and no
player will move.

Use FIFA's side-selection screen and make sure the virtual Xbox controller is on
the team you want the bot to control.

## 4. Bot Debug Overlay

In `src/main_play_against_bot.py`:

- `PAUSED`: press `p` in the debug window.
- `ctrl=0`: bot has no controlled player target.
- `stick=(0.00,0.00)`: policy is not asking for movement.
- `stick` nonzero but no movement: FIFA is not receiving the controller.

## 5. Safe Stop

Click the debug window and press:

```text
q
```

The bot resets the controller on exit.
