from pathlib import Path
from typing import Optional
import time

import cv2
import dxcam
import numpy as np

camera = dxcam.create(output_color="BGR")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
_preview_windows = set()


def make_screen(max_retries: int = 10) -> Optional[np.ndarray]:
    for _ in range(max_retries):
        frame = camera.grab()
        if frame is not None:
            return frame
        time.sleep(0.01)
    return None


def crop_image(
    image: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    if region is None:
        return image

    x1, y1, x2, y2 = region
    return image[y1:y2, x1:x2]


def load_templates(template_dir: str = "template") -> list[dict]:
    path = Path(template_dir)
    if not path.exists():
        return []

    templates = []
    for file_path in path.iterdir():
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        templates.append(
            {
                "name": file_path.stem,
                "image": image,
            }
        )

    return templates


def find_best_template(image: np.ndarray, templates: list[dict]) -> dict | None:
    best_match = None
    image_height, image_width = image.shape[:2]

    for template in templates:
        template_image = template["image"]
        template_height, template_width = template_image.shape[:2]

        if template_width > image_width or template_height > image_height:
            continue

        result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, position = cv2.minMaxLoc(result)

        if best_match is None or confidence > best_match["confidence"]:
            best_match = {
                "name": template["name"],
                "confidence": float(confidence),
                "position": position,
                "size": (template_width, template_height),
            }

    return best_match


def find_template(
    image: np.ndarray,
    templates: list[dict],
    template_name: str,
) -> dict | None:
    matching_templates = [
        template for template in templates if template["name"] == template_name
    ]
    return find_best_template(image, matching_templates)


def get_monitor_rects() -> list[tuple[int, int, int, int]]:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return []

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    monitors = []

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    monitor_enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
        ctypes.c_void_p,
    )

    def callback(monitor, _dc, rect, _data):
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(MonitorInfo)

        if ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            monitor_rect = info.rcMonitor
            is_primary = bool(info.dwFlags & 1)
        else:
            monitor_rect = rect.contents
            is_primary = False

        monitors.append(
            {
                "rect": (
                    monitor_rect.left,
                    monitor_rect.top,
                    monitor_rect.right,
                    monitor_rect.bottom,
                ),
                "is_primary": is_primary,
            }
        )
        return 1

    if not ctypes.windll.user32.EnumDisplayMonitors(
        None,
        None,
        monitor_enum_proc(callback),
        None,
    ):
        return []

    monitors.sort(
        key=lambda monitor: (
            0 if monitor["is_primary"] else 1,
            monitor["rect"][0],
            monitor["rect"][1],
        )
    )
    return [monitor["rect"] for monitor in monitors]


def create_preview_window(
    window_name: str,
    region: tuple[int, int, int, int],
    monitor_index: int = 0,
):
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    monitor_rects = get_monitor_rects()
    if 0 <= monitor_index < len(monitor_rects):
        monitor_x, monitor_y, _, _ = monitor_rects[monitor_index]
        cv2.moveWindow(window_name, monitor_x + 20, monitor_y + 20)

    _preview_windows.add(window_name)


def show_detection(
    image: np.ndarray,
    detection: dict | None,
    state_name: str,
    confidence_threshold: float,
    window_name: str = "Detection",
):
    if window_name not in _preview_windows:
        return

    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return
    except cv2.error:
        return

    preview = image.copy()

    if detection:
        x, y = detection["position"]
        width, height = detection["size"]
        confidence = detection["confidence"]
        color = (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255)

        cv2.rectangle(preview, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            preview,
            f"{confidence:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imshow(window_name, preview)
    cv2.waitKey(1)
