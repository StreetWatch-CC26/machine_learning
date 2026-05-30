# 🛣️ StreetWatch - Road Damage Detection System

StreetWatch adalah sistem berbasis Computer Vision untuk mendeteksi kerusakan jalan dari gambar. Sistem menggunakan pendekatan **multi-stage pipeline**:

1. **Road Validator** → Memastikan gambar yang diunggah benar-benar merupakan foto jalan.
2. **Pothole Detector** → Mendeteksi lubang jalan dan mengklasifikasikan tingkat keparahannya.
3. **REST API** → Menyediakan endpoint inferensi yang dapat digunakan oleh aplikasi frontend maupun mobile.

---

## 📌 Arsitektur Sistem

```text
User Upload Image
        │
        ▼
┌─────────────────────┐
│   Road Validator    │
│   MobileNetV2       │
│ (Road / Not Road)   │
└──────────┬──────────┘
           │
     Valid Road?
           │
      Yes ▼
┌─────────────────────┐
│  Pothole Detector   │
│     YOLOv8s         │
│  Severity Detection │
└──────────┬──────────┘
           │
           ▼
     JSON Response
```

> Jika gambar bukan jalan, proses dihentikan dan API mengembalikan pesan validasi.

---

## 📂 Repository Structure

```text
streetwatch/
│
├── road-validator/
│   ├── road_validator.keras
│   ├── threshold.json          ← dibaca oleh main.py
│   ├── preprocessing.py
│   └── README.md
│
├── pothole-detection-model/
│   ├── pothole_model_best.pt
│   ├── optimal_conf_threshold.txt
│   ├── predict.py
│   ├── data.yaml
│   └── README.md
│
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md
```

> **Catatan:** `main.py` membaca threshold Road Validator dari `threshold.json` (format `{"optimal_threshold": ...}`). Pastikan nama file konsisten.

---

## 🧠 Component 1: Road Validator

Model klasifikasi biner yang bertugas memverifikasi apakah gambar yang diunggah merupakan foto jalan sebelum diproses oleh model deteksi lubang.

### Model Architecture

```text
Input (224×224×3)
       ↓
Data Augmentation (Flip, Rotation, Zoom, Brightness)
       ↓
MobileNetV2 (ImageNet Pretrained)
       ↓
GlobalAveragePooling2D
       ↓
Dropout(0.3)
       ↓
Dense(128) + ReLU
       ↓
Dropout(0.3)
       ↓
Dense(1) + Sigmoid
```

### Training Strategy

**Phase 1 — Head Training**
- MobileNetV2 dibekukan (frozen)
- Epochs: 30
- Learning Rate: 0.001

**Phase 2 — Fine Tuning**
- 50 layer terakhir dibuka
- Epochs: 30
- Learning Rate: 0.0001

### Performance

| Metric            | Value  |
|-------------------|--------|
| Accuracy          | 90.48% |
| Precision (Road)  | 91.30% |
| Recall (Road)     | 89.47% |
| MAE               | 0.0952 |
| Optimal Threshold | 0.2576 |

### Confusion Matrix

```text
                Prediksi
             Road   Not Road
Road          105      10
Not Road       10      85
```

### Dataset

| Class    | Raw   | Setelah Augmentasi (used for training) |
|----------|-------|----------------------------------------|
| Road     | 600   | 800                                    |
| Not Road | 600   | 600                                    |

> Gambar jalan di-augmentasi dari 600 → 800 menggunakan flip, rotation, brightness, zoom, dan contrast. Total dataset training: 1.400 gambar (setelah split 70/15/15).

Data non-jalan berasal dari:
- Animals-10 Dataset
- Flower Classification Dataset
- Weather Image Recognition Dataset
- Internet Images (Picsum)

---

## 🕳️ Component 2: Pothole Detection

Model deteksi objek berbasis YOLOv8s yang mendeteksi lubang jalan dan mengklasifikasikan tingkat keparahannya.

### Severity Classes

| Class ID | Severity |
|----------|----------|
| 0        | Rendah   |
| 1        | Sedang   |
| 2        | Parah    |

### Dataset

**Data Split**

| Split      | Images | Annotations |
|------------|--------|-------------|
| Train      | ~665   | ~2900+      |
| Validation | 100    | 267         |

**Class Distribution**

| Severity | Count |
|----------|-------|
| Rendah   | 943   |
| Sedang   | 965   |
| Parah    | 1009  |

> Dataset relatif seimbang dengan selisih antar kelas sekitar 2%.

### Model Configuration

| Parameter  | Value   |
|------------|---------|
| Model      | YOLOv8s |
| Input Size | 640×640 |
| Epochs     | 100     |
| Batch Size | 16      |
| Optimizer  | AdamW   |
| Initial LR | 0.001   |

### Data Augmentation

```python
hsv_h = 0.02
hsv_s = 0.8
hsv_v = 0.5

degrees = 5.0
translate = 0.2
scale = 0.8

mosaic = 1.0
mixup = 0.5
```

### Detection Performance

| Metric                  | Value |
|-------------------------|-------|
| Precision               | 0.660 |
| Recall                  | 0.757 |
| F1-Score                | 0.705 |
| mAP50                   | 0.684 |
| Mean IoU                | 0.814 |
| Classification Accuracy | 89.0% |

**Per-Class Performance**

| Severity | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Rendah   | 0.96      | 0.86   | 0.91 |
| Sedang   | 0.77      | 0.86   | 0.81 |
| Parah    | 0.93      | 0.94   | 0.93 |

### Bounding Box Evaluation

| Metric       | Value    |
|--------------|----------|
| MAE Center X | 5.12 px  |
| MAE Center Y | 2.94 px  |
| MAE Width    | 11.41 px |
| MAE Height   | 8.60 px  |
| Mean IoU     | 0.814    |

### Image-Level Severity Rules

Severity keseluruhan per gambar dihitung dari semua deteksi dalam satu gambar:

| Kondisi                                   | Hasil Severity       |
|-------------------------------------------|----------------------|
| Tidak ada deteksi                         | Tidak Ada Kerusakan  |
| Ada ≥ 1 lubang kelas `parah`              | Parah                |
| Weighted score ≥ 9 (`sedang×3 + ringan×1`) | Parah               |
| Weighted score ≥ 3                        | Sedang               |
| Weighted score < 3                        | Ringan               |

### Scale Normalization

Untuk mengurangi pengaruh perspektif kamera, ukuran bounding box dinormalisasi berdasarkan estimasi jarak objek dari kamera menggunakan `ScaleNormalizer`. Hal ini membantu klasifikasi severity menjadi lebih konsisten pada kondisi jalan yang jauh dari kamera.

---

## 🌐 Component 3: API Service

REST API yang mengintegrasikan Road Validator dan Pothole Detector.

**Base URL**
```
https://firzahakim-streetwatch-model-api.hf.space
```

### Endpoints

#### `GET /` — Root Status

```json
{ "status": "ok" }
```

#### `GET /health` — Health Check Detail

```json
{
  "road_validator_loaded": true,
  "pothole_detector_loaded": true,
  "road_validator_threshold": 0.2576,
  "pothole_conf_threshold": 0.25
}
```

#### `POST /detect` — Detect Road Damage

**Request** `multipart/form-data`:
```
file=<image>
```

Supported formats: `JPG` · `JPEG` · `PNG`

### Example Responses

**Road With Damage**
```json
{
  "is_road": true,
  "total_potholes": 2,
  "image_severity": "Sedang",
  "detections": [
    {
      "bbox": [310, 184, 414, 214],
      "severity": "sedang",
      "confidence": 0.884
    }
  ]
}
```

**Normal Road (No Damage)**
```json
{
  "is_road": true,
  "total_potholes": 0,
  "image_severity": "Tidak Ada Kerusakan",
  "detections": []
}
```

**Road With Minor Damage**
```json
{
  "is_road": true,
  "total_potholes": 1,
  "image_severity": "Ringan",
  "detections": [
    {
      "bbox": [120, 90, 160, 115],
      "severity": "rendah",
      "confidence": 0.431
    }
  ]
}
```

**Not a Road**
```json
{
  "is_road": false,
  "message": "Gambar tidak terdeteksi sebagai jalan."
}
```

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Detection via Python

```python
from predict import PotholeDetector

detector = PotholeDetector(
    model_path="pothole_model_best.pt",
    conf_threshold=0.25
)

result = detector.predict("road.jpg")
print(result)
```

### Test API via Python

```python
import requests

with open("road.jpg", "rb") as f:
    response = requests.post(
        "https://firzahakim-streetwatch-model-api.hf.space/detect",
        files={"file": f}
    )

print(response.json())
```

---

## 🎯 Project Achievements

| Metric                  | Target | Achieved |
|-------------------------|--------|----------|
| Classification Accuracy | ≥ 85%  | 89.0%    |
| MAE                     | ≤ 0.02 | 0.0095   |
| F1-Score                | ≥ 0.70 | 0.705    |

---

## ⚠️ Current Limitations

- Akurasi dapat menurun pada gambar dengan resolusi sangat rendah.
- Performa dapat menurun pada kondisi malam tanpa pencahayaan memadai.
- Gambar dengan blur berat dapat memengaruhi hasil validasi maupun deteksi.
- Sudut pengambilan gambar yang sangat ekstrem belum dijamin konsisten.

---

## 🛠️ Tech Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

- **TensorFlow** + MobileNetV2
- **Ultralytics YOLOv8**
- **OpenCV** + NumPy
- **Scikit-Learn**
- **FastAPI**
- **Hugging Face Spaces**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> **StreetWatch** menggabungkan validasi gambar jalan dan deteksi lubang jalan dalam satu pipeline end-to-end untuk menghasilkan sistem analisis kerusakan jalan yang lebih robust dan siap diintegrasikan dengan aplikasi frontend maupun mobile.
