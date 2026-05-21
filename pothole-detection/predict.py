"""
Pothole Detection Inference Script

CLI : python predict.py --image foto.jpg
     python predict.py --image foto.jpg --conf 0.4

Python:
    from predict import PotholeDetector
    detector = PotholeDetector()
    result   = detector.predict("foto.jpg")
"""

import json
import argparse
from ultralytics import YOLO


def classify_image_severity(class_ids):
    num_parah  = class_ids.count(2)
    num_sedang = class_ids.count(1)
    num_ringan = class_ids.count(0)
    if num_parah == 0 and num_sedang == 0 and num_ringan == 0:
        return "Tidak Ada Kerusakan"
    if num_parah >= 1:
        return "Parah"
    score = (num_sedang * 3) + (num_ringan * 1)
    if score >= 9:
        return "Parah"
    elif score >= 3:
        return "Sedang"
    else:
        return "Ringan"


class PotholeDetector:
    def __init__(self, model_path="pothole_model_best.pt", conf_threshold=0.25):
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.classes        = ["rendah", "sedang", "parah"]

    def predict(self, image_path):
        results    = self.model(image_path, conf=self.conf_threshold)
        detections = []
        class_ids  = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                class_ids.append(cls)
                detections.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "class":      self.classes[cls],
                    "confidence": float(box.conf[0])
                })
        return {
            "success":        True,
            "num_detections": len(detections),
            "detections":     detections,
            "image_severity": classify_image_severity(class_ids)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf",  type=float, default=0.25)
    args = parser.parse_args()
    detector = PotholeDetector(conf_threshold=args.conf)
    print(json.dumps(detector.predict(args.image), indent=2))
