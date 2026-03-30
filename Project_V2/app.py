"""
app.py
Flask entry point for the Real-Time Traffic Detection System.

Endpoints:
    GET  /              → Status page (HTML dashboard)
    POST /predict       → Single image detection (JSON response)
    GET  /video_feed    → Live MJPEG webcam stream
"""

import os
import io
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, render_template_string

from utils.detector   import load_model, detect
from utils.visualizer import draw_boxes
from utils.stream     import generate_frames
from utils.database   import (
    connect as db_connect,
    get_recent_detections,
    get_traffic_analytics,
    get_peak_times,
    is_connected,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route: /  — Status dashboard
# ─────────────────────────────────────────────────────────────────────────────
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Traffic Detection API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Courier New', monospace;
      background: #0d0d0d; color: #e0e0e0;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh; padding: 2rem;
    }
    .card {
      border: 1px solid #2a2a2a; border-radius: 8px;
      padding: 2.5rem; max-width: 600px; width: 100%;
      background: #111;
    }
    h1 { color: #00d4ff; font-size: 1.4rem; margin-bottom: .5rem; }
    .badge {
      display: inline-block; background: #00d4ff22; color: #00d4ff;
      border: 1px solid #00d4ff44; padding: .2rem .7rem;
      border-radius: 99px; font-size: .75rem; margin-bottom: 2rem;
    }
    h2 { font-size: .9rem; color: #888; text-transform: uppercase;
         letter-spacing: .1em; margin: 1.5rem 0 .75rem; }
    .endpoint {
      background: #1a1a1a; border-radius: 6px;
      padding: .85rem 1rem; margin-bottom: .5rem;
      display: flex; align-items: center; gap: .75rem;
    }
    .method {
      font-size: .7rem; font-weight: bold; padding: .2rem .5rem;
      border-radius: 4px; flex-shrink: 0;
    }
    .get  { background: #0a3; color: #fff; }
    .post { background: #a50; color: #fff; }
    .path { color: #e0e0e0; font-size: .9rem; }
    .desc { color: #666; font-size: .8rem; }
    a { color: #00d4ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🚦 Traffic Detection API</h1>
    <span class="badge">● online</span>

    <h2>Endpoints</h2>

    <div class="endpoint">
      <span class="method get">GET</span>
      <div>
        <div class="path">/</div>
        <div class="desc">This status page</div>
      </div>
    </div>

    <div class="endpoint">
      <span class="method post">POST</span>
      <div>
        <div class="path">/predict</div>
        <div class="desc">Upload an image → receive JSON detections</div>
      </div>
    </div>

    <div class="endpoint">
      <span class="method get">GET</span>
      <div>
        <div class="path"><a href="/video_feed">/video_feed</a></div>
        <div class="desc">Live annotated webcam stream (MJPEG)</div>
      </div>
    </div>

    <h2>Model Classes</h2>
    <div class="endpoint">
      <span style="color:#aaa;font-size:.85rem">
        person &nbsp;·&nbsp; bicycle &nbsp;·&nbsp; car &nbsp;·&nbsp;
        bus &nbsp;·&nbsp; motorbike
      </span>
    </div>
  </div>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Route: POST /predict — single image detection
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict():
    """
    Accept an image file and return object detections as JSON.

    Request:
        multipart/form-data with field: "image"

    Response 200:
        {
            "objects": [
                {"label": "car", "confidence": 0.92, "bbox": [x1,y1,x2,y2]},
                ...
            ],
            "count": 3
        }

    Response 400 / 500:
        { "error": "<message>" }
    """
    # ── Validate request ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        }), 400

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img_bytes = file.read()
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        image     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image. Ensure it is a valid image file."}), 400

    except Exception as e:
        logger.error(f"[/predict] Image decode error: {e}")
        return jsonify({"error": "Failed to read image."}), 400

    # ── Run detection ─────────────────────────────────────────────────────────
    try:
        detections = detect(image)
    except Exception as e:
        logger.error(f"[/predict] Detection error: {e}")
        return jsonify({"error": "Inference failed. Please try again."}), 500

    # ── Format response ───────────────────────────────────────────────────────
    objects = [
        {
            "label":      d["label"],
            "confidence": d["conf"],
            "bbox":       d["bbox"],
        }
        for d in detections
    ]

    return jsonify({"objects": objects, "count": len(objects)})


# ─────────────────────────────────────────────────────────────────────────────
# Route: GET /detections — fetch stored detections from MongoDB
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/detections")
def detections_list():
    """
    Fetch recent detection records from MongoDB.

    Query params:
        limit      (int):  Max records to return (default 50, max 200)
        start_time (str):  ISO 8601 lower bound  e.g. 2026-01-10T00:00:00Z
        end_time   (str):  ISO 8601 upper bound

    Response 200:
        [
            {
                "timestamp":     "2026-01-10T14:28:01Z",
                "source":        "video",
                "total_objects": 6,
                "detections":    [{"label": "car", "confidence": 0.89, "bbox": [...]}]
            }
        ]
    """
    if not is_connected():
        return jsonify({"error": "Database unavailable."}), 503

    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "'limit' must be an integer."}), 400

    start_time = end_time = None
    try:
        from datetime import datetime
        raw_start = request.args.get("start_time")
        raw_end   = request.args.get("end_time")
        if raw_start:
            start_time = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
        if raw_end:
            end_time = datetime.fromisoformat(raw_end.replace("Z", "+00:00"))
    except ValueError as e:
        return jsonify({"error": f"Invalid datetime format: {e}"}), 400

    docs = get_recent_detections(limit=limit, start_time=start_time, end_time=end_time)
    for doc in docs:
        if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
            doc["timestamp"] = doc["timestamp"].isoformat()

    return jsonify(docs)


# ─────────────────────────────────────────────────────────────────────────────
# Route: GET /analytics/traffic — object class distribution
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/analytics/traffic")
def analytics_traffic():
    """
    Total detection count per object class across all stored frames.

    Response 200:
        { "car": 342, "person": 120, "bus": 45, "bicycle": 30, "motorbike": 18 }
    """
    if not is_connected():
        return jsonify({"error": "Database unavailable."}), 503

    return jsonify(get_traffic_analytics())


# ─────────────────────────────────────────────────────────────────────────────
# Route: GET /analytics/peak-time — busiest hours of day
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/analytics/peak-time")
def analytics_peak_time():
    """
    Total detected objects grouped by hour-of-day (UTC), busiest first.

    Response 200:
        [
            { "hour": 8,  "total_objects": 1240, "frame_count": 310 },
            { "hour": 17, "total_objects": 980,  "frame_count": 245 }
        ]
    """
    if not is_connected():
        return jsonify({"error": "Database unavailable."}), 503

    return jsonify(get_peak_times())


# ─────────────────────────────────────────────────────────────────────────────
# Route: GET /video_feed — live MJPEG stream
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/video_feed")
def video_feed():
    """
    Streams annotated webcam frames as MJPEG.
    Open in browser or embed in an <img> tag:
        <img src="/video_feed">
    """
    source = request.args.get("source", default=0)

    # Allow numeric webcam index via query param: /video_feed?source=1
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass   # keep as string (file path / URL)

    return Response(
        generate_frames(source),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("[app] 🔄 Loading YOLOv8 model at startup …")
    load_model()   # Eager load — avoids cold-start on first request

    logger.info("[app] 🔄 Connecting to MongoDB Atlas …")
    if db_connect():
        logger.info("[app] ✅ Database connected.")
    else:
        logger.warning("[app] ⚠️  Database unavailable — running without persistence.")

    logger.info("[app] 🚀 Starting Flask server on http://0.0.0.0:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,       # Set True during development
        threaded=True,     # Needed for concurrent streaming + predict calls
    )
