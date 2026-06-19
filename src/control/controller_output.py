import vgamepad as vg


MAX_THUMB_VALUE = 32767
MIN_THUMB_VALUE = -32768


class XboxController:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.reset()

    def apply_action(self, action: dict):
        left_x = self._axis_to_thumb(action.get("left_stick_x", 0.0))
        # Image coordinates have positive Y downward; Xbox stick Y is positive upward.
        left_y = self._axis_to_thumb(-action.get("left_stick_y", 0.0))

        self.gamepad.left_joystick(x_value=left_x, y_value=left_y)
        self._set_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A, action.get("press_a", False))
        self._set_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B, action.get("press_b", False))
        self._set_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X, action.get("press_x", False))
        self._set_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y, action.get("press_y", False))

        if action.get("press_rt", False):
            self.gamepad.right_trigger(value=255)
        else:
            self.gamepad.right_trigger(value=0)

        self.gamepad.update()

    def reset(self):
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.right_trigger(value=0)

        for button in (
            vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        ):
            self.gamepad.release_button(button=button)

        self.gamepad.update()

    def _set_button(self, button, pressed: bool):
        if pressed:
            self.gamepad.press_button(button=button)
        else:
            self.gamepad.release_button(button=button)

    def _axis_to_thumb(self, value: float) -> int:
        value = max(-1.0, min(1.0, float(value)))
        if value < 0:
            return int(value * abs(MIN_THUMB_VALUE))
        return int(value * MAX_THUMB_VALUE)
