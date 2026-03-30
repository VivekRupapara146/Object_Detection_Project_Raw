"""
database.py
MongoDB Atlas integration layer.

Responsibilities:
  - Singleton MongoClient (one connection pool for the entire app)
  - Non-blocking async writes via a background thread queue
  - Insert frame detections into `frames` collection
  - Query helpers for /detections and /analytics endpoints
  - Graceful degradation — if DB is down, system keeps running
"""

import os
import logging
import threading
import queue
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Replace the placeholder below with your Atlas connection string.
# Recommended: load from environment variable in production.
#
#   export MONGO_URI="mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
#
MONGO_URI       = os.getenv("MONGO_URI", "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority")
DB_NAME         = os.getenv("DB_NAME",   "traffic_detection")
COLLECTION_FRAMES = "frames"
COLLECTION_USERS  = "users"

# ── Write strategy thresholds ─────────────────────────────────────────────────
SAVE_EVERY_N_FRAMES    = 5      # Save 1 out of every 5 frames
MIN_CONFIDENCE_TO_SAVE = 0.5    # Only save if at least one detection >= this

# ── Singleton state ───────────────────────────────────────────────────────────
_client: MongoClient | None = None
_db                          = None
_write_queue: queue.Queue    = queue.Queue(maxsize=500)   # bounded to prevent memory bloat
_worker_thread: threading.Thread | None = None
_frame_counter: int          = 0
_counter_lock                = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def connect() -> bool:
    """
    Initialise the MongoClient singleton and ensure indexes exist.
    Called once at app startup.

    Returns:
        bool: True if connected successfully, False otherwise.
    """
    global _client, _db

    if _client is not None:
        return True

    try:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5_000,   # 5 s connection timeout
            connectTimeoutMS=5_000,
            socketTimeoutMS=10_000,
            maxPoolSize=10,
        )
        # Force a round-trip to verify the connection
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        _ensure_indexes()
        _start_write_worker()
        logger.info(f"[database] ✅ Connected to MongoDB Atlas — db: '{DB_NAME}'")
        return True

    except ConnectionFailure as e:
        logger.error(f"[database] ❌ MongoDB connection failed: {e}")
        _client = None
        _db     = None
        return False

    except Exception as e:
        logger.error(f"[database] ❌ Unexpected error during connect: {e}")
        _client = None
        _db     = None
        return False


def is_connected() -> bool:
    return _client is not None and _db is not None


def _ensure_indexes():
    """Create indexes on first connect (idempotent)."""
    try:
        frames = _db[COLLECTION_FRAMES]
        # Primary query field — time-range lookups
        frames.create_index([("timestamp", DESCENDING)], name="idx_timestamp")
        # Analytics grouping
        frames.create_index([("timestamp", ASCENDING), ("detections.label", ASCENDING)],
                            name="idx_timestamp_label")
        # Optional TTL: auto-delete docs older than 30 days
        # frames.create_index("timestamp", expireAfterSeconds=2_592_000, name="ttl_30d")
        logger.info("[database] ✅ Indexes ensured.")
    except Exception as e:
        logger.warning(f"[database] Index creation warning: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Async Write Worker
# ─────────────────────────────────────────────────────────────────────────────

def _start_write_worker():
    """Spawn a daemon thread that drains the write queue in the background."""
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        return

    _worker_thread = threading.Thread(target=_write_loop, daemon=True, name="db-writer")
    _worker_thread.start()
    logger.info("[database] 🧵 Background write worker started.")


def _write_loop():
    """Continuously drain documents from the queue and insert them."""
    while True:
        try:
            doc = _write_queue.get(timeout=1)   # blocks until item available
            if doc is None:                      # sentinel → shutdown signal
                break
            _db[COLLECTION_FRAMES].insert_one(doc)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[database] Write worker insert error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Write Strategy: Combined (every-N + min-confidence)
# ─────────────────────────────────────────────────────────────────────────────

def should_save(detections: list[dict]) -> bool:
    """
    Combined write strategy:
      1. Frame counter must be a multiple of SAVE_EVERY_N_FRAMES
      2. At least one detection must meet MIN_CONFIDENCE_TO_SAVE

    Both conditions must be true.
    """
    global _frame_counter

    with _counter_lock:
        _frame_counter += 1
        save_by_count = (_frame_counter % SAVE_EVERY_N_FRAMES == 0)

    if not save_by_count:
        return False

    if not detections:
        return False

    save_by_conf = any(d.get("conf", 0) >= MIN_CONFIDENCE_TO_SAVE for d in detections)
    return save_by_conf


def save_frame(detections: list[dict], source: str = "video") -> bool:
    """
    Enqueue a frame document for async insertion into MongoDB.
    Returns immediately — does NOT block the inference pipeline.

    Args:
        detections (list[dict]): Output from detector.detect()
        source     (str):        "video" | "image"

    Returns:
        bool: True if enqueued, False if DB unavailable or queue full.
    """
    if not is_connected():
        return False

    if not should_save(detections):
        return False

    doc = {
        "timestamp":     datetime.now(timezone.utc),
        "source":        source,
        "total_objects": len(detections),
        "detections": [
            {
                "label":      d["label"],
                "confidence": d["conf"],
                "bbox":       d["bbox"],
            }
            for d in detections
        ],
    }

    try:
        _write_queue.put_nowait(doc)
        return True
    except queue.Full:
        logger.warning("[database] Write queue full — frame dropped.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Query Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_detections(limit: int = 50,
                          start_time: datetime = None,
                          end_time:   datetime = None) -> list[dict]:
    """
    Fetch recent frame documents from the `frames` collection.

    Args:
        limit      (int):      Max number of docs to return (capped at 200).
        start_time (datetime): Optional lower bound on timestamp.
        end_time   (datetime): Optional upper bound on timestamp.

    Returns:
        list[dict]: Serialisable list of frame documents.
    """
    if not is_connected():
        return []

    limit = min(limit, 200)
    query = {}

    if start_time or end_time:
        query["timestamp"] = {}
        if start_time:
            query["timestamp"]["$gte"] = start_time
        if end_time:
            query["timestamp"]["$lte"] = end_time

    try:
        cursor = (
            _db[COLLECTION_FRAMES]
            .find(query, {"_id": 0})          # exclude ObjectId from response
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"[database] get_recent_detections error: {e}")
        return []


def get_traffic_analytics() -> dict:
    """
    Aggregate object-type distribution across all stored frames.

    Returns:
        dict: { "car": 342, "person": 120, ... }
    """
    if not is_connected():
        return {}

    pipeline = [
        {"$unwind": "$detections"},
        {"$group": {
            "_id":   "$detections.label",
            "count": {"$sum": 1},
        }},
        {"$sort": {"count": DESCENDING}},
    ]

    try:
        results = _db[COLLECTION_FRAMES].aggregate(pipeline)
        return {r["_id"]: r["count"] for r in results}
    except Exception as e:
        logger.error(f"[database] get_traffic_analytics error: {e}")
        return {}


def get_peak_times() -> list[dict]:
    """
    Group detections by hour-of-day to identify peak traffic windows.

    Returns:
        list[dict]: [{ "hour": 8, "total_objects": 1240 }, ...]
                    Sorted by total_objects descending.
    """
    if not is_connected():
        return []

    pipeline = [
        {"$group": {
            "_id": {
                "hour": {"$hour": "$timestamp"},
            },
            "total_objects": {"$sum": "$total_objects"},
            "frame_count":   {"$sum": 1},
        }},
        {"$project": {
            "_id":          0,
            "hour":         "$_id.hour",
            "total_objects": 1,
            "frame_count":  1,
        }},
        {"$sort": {"total_objects": DESCENDING}},
    ]

    try:
        return list(_db[COLLECTION_FRAMES].aggregate(pipeline))
    except Exception as e:
        logger.error(f"[database] get_peak_times error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# User helpers (Auth scaffolding)
# ─────────────────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str) -> bool:
    """Insert a new user document. Returns False if email already exists."""
    if not is_connected():
        return False
    try:
        _db[COLLECTION_USERS].insert_one({
            "username":      username,
            "email":         email,
            "password_hash": password_hash,
            "created_at":    datetime.now(timezone.utc),
        })
        return True
    except Exception as e:
        logger.error(f"[database] create_user error: {e}")
        return False


def find_user_by_email(email: str) -> dict | None:
    """Return user document by email, or None if not found."""
    if not is_connected():
        return None
    try:
        return _db[COLLECTION_USERS].find_one({"email": email}, {"_id": 0})
    except Exception as e:
        logger.error(f"[database] find_user_by_email error: {e}")
        return None
