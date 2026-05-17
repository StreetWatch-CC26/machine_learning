# Pothole Detection Model — YOLOv8s

## Deskripsi
Model deteksi lubang jalan dengan 3 tingkat keparahan: **Rendah**, **Sedang**, **Parah**.

---

## Performa Model

| Metrik                | Nilai  |
|-----------------------|--------|
| Precision             | 66.0%  |
| Recall                | 75.7%  |
| F1-Score              | 70.5%  |
| mAP50                 | 68.4%  |
| Akurasi Klasifikasi   | 89.0%  |
| Mean IoU              | 81.4%  |

### Per-Class Performance

| Kelas  | Precision | Recall | F1   |
|--------|-----------|--------|------|
| rendah | 96%       | 86%    | 91%  |
| sedang | 77%       | 86%    | 81%  |
| parah  | 93%       | 94%    | 93%  |

---

## Cara Penggunaan

### Install dependencies
```bash
pip install -r requirements.txt
```

### Inferensi via CLI
```bash
python predict.py --image foto.jpg
python predict.py --image foto.jpg --conf 0.4
```

### Inferensi via Python
```python
from predict import PotholeDetector

detector = PotholeDetector(model_path="pothole_model_best.pt", conf_threshold=0.25)
result   = detector.predict("foto.jpg")
print(result)
```

### Contoh output
```json
{
  "success": true,
  "num_detections": 2,
  "detections": [
    {"bbox": [120, 80, 340, 210], "class": "parah",  "confidence": 0.91},
    {"bbox": [400, 150, 560, 280],"class": "rendah", "confidence": 0.67}
  ]
}
```

---

## File Package

| File                         | Keterangan                     |
|------------------------------|--------------------------------|
| `pothole_model_best.pt`      | Bobot model terbaik (YOLOv8s)  |
| `predict.py`                 | Script inferensi siap pakai    |
| `data.yaml`                  | Konfigurasi dataset            |
| `model_metadata.json`        | Metadata & performa model      |
| `optimal_conf_threshold.txt` | Threshold confidence optimal   |
| `requirements.txt`           | Dependensi Python              |

---

## Training Info
- **Arsitektur**: YOLOv8s
- **Epochs**: 100  |  **Batch**: 16  |  **Imgsz**: 640
- **Optimizer**: AdamW
- **Confidence Threshold**: 0.25
