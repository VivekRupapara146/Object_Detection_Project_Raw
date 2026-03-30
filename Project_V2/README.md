# 🚦 Real-Time Traffic Detection System

YOLOv8 + Flask backend for real-time traffic object detection.

---

## Project Structure

```
project/
├── app.py                  # Flask entry point
├── requirements.txt
├── model/
│   └── best.pt             # ← Place your YOLOv8 model here
└── utils/
    ├── __init__.py
    ├── detector.py         # Inference logic (singleton model)
    ├── visualizer.py       # Bounding box drawing
    └── stream.py           # Webcam capture & MJPEG streaming
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place your model**
```
model/best.pt
```

**3. Set your MongoDB Atlas connection string**
```bash
export MONGO_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
```
Or edit `utils/database.py` and replace the `MONGO_URI` placeholder directly.

**4. Run the server**
```bash
python app.py
```

Server starts at `http://localhost:5000`

---

## API Endpoints

### `GET /`
Status dashboard — lists all available endpoints.

---

### `POST /predict`
Upload an image and receive structured detection results.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/your/image.jpg"
```

**Response:**
```json
{
  "objects": [
    { "label": "car",    "confidence": 0.92, "bbox": [120, 80, 340, 220] },
    { "label": "person", "confidence": 0.87, "bbox": [400, 60, 460, 210] }
  ],
  "count": 2
}
```

---

### `GET /video_feed`
Returns a live MJPEG stream from your webcam, annotated with detections.

**In browser:** open `http://localhost:5000/video_feed`

**In HTML:**
```html
<img src="http://localhost:5000/video_feed" />
```

**Switch webcam:**
```
http://localhost:5000/video_feed?source=1
```

**Use a video file:**
```
http://localhost:5000/video_feed?source=/path/to/video.mp4
```

---

### `GET /detections`
Fetch stored detection records from MongoDB.
```bash
curl "http://localhost:5000/detections?limit=20"
curl "http://localhost:5000/detections?start_time=2026-01-10T00:00:00Z"
```

---

### `GET /analytics/traffic`
Object class distribution across all stored frames.
```bash
curl http://localhost:5000/analytics/traffic
# { "car": 342, "person": 120, "bus": 45, "bicycle": 30, "motorbike": 18 }
```

---

### `GET /analytics/peak-time`
Busiest hours of day ranked by total detection volume.
```bash
curl http://localhost:5000/analytics/peak-time
# [{ "hour": 8, "total_objects": 1240, "frame_count": 310 }, ...]
```

---

## Detectable Classes

| Class      | Color     |
|------------|-----------|
| person     | Amber     |
| bicycle    | Green     |
| car        | Blue      |
| bus        | Purple    |
| motorbike  | Orange    |

---

## Configuration

| Parameter             | File          | Default |
|-----------------------|---------------|---------|
| Confidence threshold  | detector.py   | 0.3     |
| Stream resolution     | stream.py     | 640×480 |
| Target FPS            | stream.py     | 30      |
| JPEG quality          | stream.py     | 85      |
| Max upload size       | app.py        | 16 MB   |
| Save every N frames   | database.py   | 5       |
| Min save confidence   | database.py   | 0.5     |
| Write queue max size  | database.py   | 500     |

---

## Testing

**Test detector independently (no Flask needed):**
```python
import cv2
from utils.detector import detect

img = cv2.imread("test.jpg")
results = detect(img)
print(results)
```

**Test visualizer:**
```python
import cv2
from utils.detector   import detect
from utils.visualizer import draw_boxes

img        = cv2.imread("test.jpg")
detections = detect(img)
annotated  = draw_boxes(img, detections)
cv2.imwrite("output.jpg", annotated)
```

---

## Future Extensions

- [ ] Object tracking with DeepSORT
- [ ] Vehicle counting per ROI
- [ ] Speed estimation
- [ ] Docker deployment
- [ ] RTSP / IP camera support
