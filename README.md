# PKR Currency Note Detection 🏦💵

A **YOLOv8-based deep learning model** for detecting and classifying **Pakistani currency notes** (PKR 10, 20, 50, 100, 500, 1000, 5000) with high accuracy. The project includes dataset preparation, model training, validation, and inference for real-time banknote detection.

---

## 🚀 Features

* Detects and classifies **7 denominations of PKR notes**
* Trained on a **custom dataset** using YOLOv8
* Achieved **mAP50 ≈ 98.5%** and **mAP50-95 ≈ 92.7%**
* Supports **training, validation, and inference**
* Works with both **images and datasets**

---

## 📂 Dataset

* Custom dataset annotated in **YOLO format**
* Classes:

  * `pkr-10`
  * `pkr-20`
  * `pkr-50`
  * `pkr-100`
  * `pkr-500`
  * `pkr-1000`
  * `pkr-5000`
* Dataset split into **train** and **valid**

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/pkr-currency-detection.git
cd pkr-currency-detection
pip install ultralytics matplotlib
```

---

## 📌 Training

```bash
yolo detect train \
  data="data.yaml" \
  model=yolov8n.pt \
  epochs=150 \
  name="train_pkr" \
  lr0=0.01 \
  lrf=0.1 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=3 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  box=7.5 \
  cls=0.5 \
  dfl=1.0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  translate=0.1 \
  scale=0.5 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.2
```

---

## 📊 Results

| Metric    | Value |
| --------- | ----- |
| Precision | 95.1% |
| Recall    | 96.8% |
| mAP@50    | 98.5% |
| mAP@50-95 | 92.7% |

---

## 🔎 Inference

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("runs/detect/train_pkr/weights/best.pt")
results = model("path/to/test_image.jpg")

res = results[0]
im = res.plot()
plt.imshow(im)
plt.axis("off")
plt.show()
```

---

## 📘 Notebook

The full training, validation, and inference pipeline is available in:
**`Pkr-Currency-Notes.ipynb`**

---

## 📌 Future Work

* Increase dataset size for improved generalization
* Deploy as a **web or mobile application**
* Extend model for **counterfeit note detection**

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* Custom PKR currency dataset

