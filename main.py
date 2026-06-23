import time
import random
import cv2
import pydirectinput

from funcs import (
    create_preview_window,
    crop_image,
    find_template,
    load_templates,
    make_screen,
    show_detection,
)
from state_machine import StateMachine

TEMPLATE_DIR = "template"
TEMPLATE_NAME = "autopilot_off"
CROP_REGION = (115, 1325, 270, 1390)  # TODO: set final coordinates later
PREVIEW_WINDOW_NAME = "Detection"
PREVIEW_MONITOR_INDEX = 1  # 0 - main monitor, 1 - second monitor
LOOP_DELAY = 0.05
TEMPLATE_RELOAD_SECONDS = 1.0

pydirectinput.PAUSE = 0


def main():
    state_machine = StateMachine(template_name=TEMPLATE_NAME)
    templates = []
    last_template_reload = 0
    last_detection_log = 0
    last_state = state_machine.state

    create_preview_window(PREVIEW_WINDOW_NAME, CROP_REGION, PREVIEW_MONITOR_INDEX)

    while True:
        now = time.time()
        if now - last_template_reload >= TEMPLATE_RELOAD_SECONDS:
            templates = load_templates(TEMPLATE_DIR)
            last_template_reload = now
            if not any(template["name"] == TEMPLATE_NAME for template in templates):
                print(f"Template not loaded: {TEMPLATE_NAME}.png")

        screen = make_screen()
        if screen is None:
            continue

        cropped = crop_image(screen, CROP_REGION)
        detection = find_template(cropped, templates, TEMPLATE_NAME)
        state = state_machine.update(detection)

        if now - last_detection_log >= 1:
            # print_status(state.name, detection)
            last_detection_log = now

        show_detection(
            cropped,
            detection,
            state.name,
            state_machine.confidence_threshold,
            PREVIEW_WINDOW_NAME,
        )

        if state != last_state:
            # print_status(state.name, detection)
            if state.name == "TEMPLATE_SEEN":
                handle_template_seen()
            last_state = state

        human_press("enter")


def human_press(key: str):
    time.sleep(random.uniform(0.04, 0.18))
    pydirectinput.keyDown(key)
    time.sleep(random.uniform(0.03, 0.11))
    pydirectinput.keyUp(key)
    time.sleep(random.uniform(0.08, 0.24))


def handle_template_seen():
    human_press("c")
    time.sleep(random.uniform(0.5, 0.99))
    human_press("2")
    time.sleep(random.uniform(0.5, 0.99))


def print_status(state_name: str, detection: dict | None):
    if detection:
        print(
            f"{state_name}: {detection['name']} "
            f"{detection['confidence']:.3f} at {detection['position']}"
        )
    else:
        print(state_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Stopped")
