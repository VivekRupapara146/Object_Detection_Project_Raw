"""
visualizer.py
Draws bounding boxes, labels, and confidence scores onto frames.
Fully independent of Flask or the model — pure OpenCV utility.
"""

import cv2
import numpy as np

# ── Color palette per class ───────────────────────────────────────────────────
# BGR format
CLASS_COLORS = {
    "person":    (0,   200, 255),   # amber
    "bicycle":   (0,   255, 128),   # green-mint
    "car":       (255, 80,  80 ),   # blue
    "bus":       (128, 0,   255),   # purple
    "motorbike": (0,   128, 255),   # orange
}
DEFAULT_COLOR = (200, 200, 200)     # grey fallback


def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Annotate a BGR image with bounding boxes, class labels, and confidence.

    Args:
        image      (np.ndarray):  BGR frame to annotate (modified in-place).
        detections (list[dict]):  Output from detector.detect().

    Returns:
        np.ndarray: Annotated image.
    """
    if image is None:
        raise ValueError("draw_boxes received a None image.")

    for det in detections:
        label = det.get("label", "unknown")
        conf  = det.get("conf",  0.0)
        bbox  = det.get("bbox",  [])

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # ── Label background ──────────────────────────────────────────────────
        text  = f"{label} {conf:.0%}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

        # Keep label inside frame boundaries
        label_y = max(y1 - 6, th + baseline)
        cv2.rectangle(
            image,
            (x1, label_y - th - baseline),
            (x1 + tw + 4, label_y + baseline),
            color,
            thickness=cv2.FILLED
        )

        # ── Label text ────────────────────────────────────────────────────────
        cv2.putText(
            image, text,
            (x1 + 2, label_y),
            font, scale,
            (0, 0, 0),   # black text on coloured bg
            thick, cv2.LINE_AA
        )

    return image
