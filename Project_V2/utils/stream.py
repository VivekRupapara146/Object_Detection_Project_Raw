"""
stream.py
Real-time video capture, inference, and MJPEG frame generation.
Designed to be consumed by Flask's streaming response.
"""

import cv2
import logging
from utils.detector   import detect
from utils.visualizer import draw_boxes

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30


def _open_capture(source) -> cv2.VideoCapture:
    """
    Open a VideoCapture from webcam index (int) or file/URL (str).
    Raises RuntimeError if the source cannot be opened.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source!r}. "
            "Check camera permissions or file path."
        )
    # Hint the driver about desired resolution / FPS (not guaranteed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    return cap


def generate_frames(source=0):
    """
    Generator that yields MJPEG-encoded bytes for Flask streaming.

    Each yielded chunk is a complete JPEG frame wrapped in
    multipart/x-mixed-replace boundary format.

    Args:
        source: Webcam index (default 0) or path to video file / RTSP URL.

    Yields:
        bytes: Encoded JPEG frame ready for HTTP streaming.
    """
    try:
        cap = _open_capture(source)
    except RuntimeError as e:
        logger.error(str(e))
        return   # stop the generator gracefully; Flask will close the response

    logger.info(f"[stream] 🎥 Streaming started from source: {source!r}")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                logger.warning("[stream] ⚠️  Failed to read frame — stream may have ended.")
                break

            # ── Optional resize for consistent inference speed ────────────────
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ── Detection + annotation ────────────────────────────────────────
            try:
                detections = detect(frame)
                frame      = draw_boxes(frame, detections)
            except Exception as e:
                logger.error(f"[stream] Inference error: {e}")
                # Continue streaming even if one frame fails

            # ── JPEG encoding ─────────────────────────────────────────────────
            success, buffer = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]   # quality/speed trade-off
            )
            if not success:
                continue

            # ── Yield MJPEG chunk ─────────────────────────────────────────────
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        cap.release()
        logger.info("[stream] 🛑 Camera released.")
