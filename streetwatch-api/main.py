import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).parent

ROAD_VALIDATOR_MODEL_PATH     = BASE_DIR / "road-validator" / "road_validator.keras"
ROAD_VALIDATOR_THRESHOLD_PATH = BASE_DIR / "road-validator" / "threshold.txt"
POTHOLE_MODEL_PATH            = BASE_DIR / "pothole-detection-model" / "pothole_model_best.pt"
POTHOLE_THRESHOLD_PATH        = BASE_DIR / "pothole-detection-model" / "optimal_conf_threshold.txt"
POTHOLE_CLASSES = ["rendah", "sedang", "parah"]
INPUT_SIZE = (224, 224)

road_validator_model = None
road_validator_threshold = None
pothole_model = None
pothole_conf_threshold = None


def load_models():
    global road_validator_model, road_validator_threshold
    global pothole_model, pothole_conf_threshold

    road_validator_model = tf.keras.models.load_model(str(ROAD_VALIDATOR_MODEL_PATH))

    with open(ROAD_VALIDATOR_THRESHOLD_PATH) as f:
        road_validator_threshold = json.load(f)["optimal_threshold"]

    pothole_model = YOLO(str(POTHOLE_MODEL_PATH))

    with open(POTHOLE_THRESHOLD_PATH) as f:
        pothole_conf_threshold = float(f.read().strip())


def classify_image_severity(class_ids: list[int]) -> str:
    if not class_ids:
        return "Tidak Ada Kerusakan"

    if class_ids.count(2) >= 1:
        return "Parah"

    score = (class_ids.count(1) * 3) + class_ids.count(0)
    if score >= 9:
        return "Parah"
    elif score >= 3:
        return "Sedang"
    else:
        return "Ringan"


app = FastAPI(title="Road Damage Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {
        "road_validator_loaded": road_validator_model is not None,
        "pothole_detector_loaded": pothole_model is not None,
        "road_validator_threshold": road_validator_threshold,
        "pothole_conf_threshold": pothole_conf_threshold,
    }


@app.post("/detect")
async def detect_road_damage(file: UploadFile = File(...)):
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="File tidak dapat dibaca sebagai gambar.")

    # Road Validator
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    road_prob = float(road_validator_model.predict(img, verbose=0)[0][0])

    if road_prob < road_validator_threshold:
        return {
            "is_road": False,
            "message": "Gambar tidak terdeteksi sebagai jalan.",
        }

    # Pothole Detection
    results = pothole_model(image_bgr, conf=pothole_conf_threshold, verbose=False)

    detections = []
    class_ids = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        class_ids.append(cls_id)
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "severity": POTHOLE_CLASSES[cls_id],
            "confidence": round(float(box.conf[0]), 4),
        })

    return {
        "is_road": True,
        "total_potholes": len(detections),
        "image_severity": classify_image_severity(class_ids),
        "detections": detections,
    }
