"""
Pothole Detection Inference Script

Cara penggunaan (CLI):
    python predict.py --image foto.jpg

Cara penggunaan (Python):
    from predict import PotholeDetector
    detector = PotholeDetector()
    result  = detector.predict("foto.jpg")
"""

import json
import argparse
from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path="pothole_model_best.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.classes = ["rendah", "sedang", "parah"]

    def predict(self, image_path):
        results = self.model(image_path, conf=self.conf_threshold)
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": self.classes[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })
        return {
            "success": True,
            "num_detections": len(detections),
            "detections": detections
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--conf",  type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    detector = PotholeDetector(conf_threshold=args.conf)
    result   = detector.predict(args.image)
    print(json.dumps(result, indent=2))
