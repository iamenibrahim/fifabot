import time

from controller_output import XboxController


def hold(action, seconds=1.0):
    controller = XboxController()
    try:
        print(f"Applying action for {seconds:.1f}s: {action}")
        controller.apply_action(action)
        time.sleep(seconds)
    finally:
        controller.reset()
        print("Controller reset.")


def main():
    print("Virtual Xbox controller input test")
    print("Open FIFA menus or practice arena and watch whether input is received.")
    input("Press Enter to move LEFT stick right for 2 seconds...")
    hold(
        {
            "left_stick_x": 1.0,
            "left_stick_y": 0.0,
            "press_a": False,
            "press_b": False,
            "press_x": False,
            "press_y": False,
            "press_rt": False,
        },
        seconds=2.0,
    )

    input("Press Enter to hold RT and move stick up for 2 seconds...")
    hold(
        {
            "left_stick_x": 0.0,
            "left_stick_y": -1.0,
            "press_a": False,
            "press_b": False,
            "press_x": False,
            "press_y": False,
            "press_rt": True,
        },
        seconds=2.0,
    )

    input("Press Enter to press B for 0.5 seconds...")
    hold(
        {
            "left_stick_x": 0.0,
            "left_stick_y": 0.0,
            "press_a": False,
            "press_b": True,
            "press_x": False,
            "press_y": False,
            "press_rt": False,
        },
        seconds=0.5,
    )

    print("Done.")


if __name__ == "__main__":
    main()
