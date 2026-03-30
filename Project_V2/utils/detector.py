"""
detector.py
Core inference logic using YOLOv8.

Changes from v1:
  - After each detection, triggers a non-blocking DB save via database.save_frame()
  - DB failure never affects inference — fully decoupled
"""

from ultralytics import YOLO
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# ── Singleton model instance ──────────────────────────────────────────────────
_model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
CONFIDENCE_THRESHOLD = 0.3

CLASS_NAMES = ["person", "bicycle", "car", "bus", "motorbike"]


def load_model():
    """Load YOLOv8 model once and cache globally. Called at app startup."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}\n"
                "Please place best.pt inside the /model directory."
            )
        _model = YOLO(MODEL_PATH)
        logger.info(f"[detector] ✅ Model loaded from: {MODEL_PATH}")
    return _model


def get_model():
    global _model
    if _model is None:
        load_model()
    return _model


def detect(image: np.ndarray, source: str = "video") -> list[dict]:
    """
    Run YOLOv8 inference on a single BGR image (numpy array).
    Triggers a non-blocking DB write after inference.

    Args:
        image  (np.ndarray): Input frame in BGR format (from OpenCV).
        source (str):        "video" | "image" — passed to DB document.

    Returns:
        list[dict]: List of detections:
            {
                "label": str,
                "conf":  float,
                "bbox":  [x1, y1, x2, y2]
            }
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input: expected a numpy ndarray image.")

    model = get_model()
    results = model(image, verbose=False)[0]

    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        label  = results.names.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "conf":  round(conf, 4),
            "bbox":  [round(x1), round(y1), round(x2), round(y2)]
        })

    # ── Non-blocking DB write ─────────────────────────────────────────────────
    # Imported here to avoid circular imports at module level
    try:
        from utils.database import save_frame
        save_frame(detections, source=source)
    except Exception as e:
        # DB errors must NEVER crash inference
        logger.warning(f"[detector] DB save skipped: {e}")

    return detections
