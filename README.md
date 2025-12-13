# military-object-detection-yolov8
Military Object Detection Using YOLOv8  A deep-learning project to detect 12 classes of military &amp; civilian assets using the YOLOv8 object detection model. This project was developed as part of the IIT BHU Hackathon challenge.
📌 Project Overview

This repository contains:

A YOLOv8 training pipeline (Google Colab compatible)

Dataset configuration (12 classes)

Auto-generated data.yaml

Minimal one-cell training script

Instructions for inference & exporting predictions

Model weights generated from training

The project uses the Military Object Dataset (12 Classes) formatted in YOLOv8 style.

📂 Folder Structure
project/
│── data.yaml              # Auto-generated dataset configuration
│── notebook.ipynb         # Colab notebook connected to GitHub
│── runs/                  # YOLO training output (after running training)
│── README.md              # Project documentation


Your dataset structure (stored in Google Drive):

military_object_dataset/
│── train/
│   ├── images/
│   └── labels/
│── val/
│   ├── images/
│   └── labels/
│── test/
│   └── images/
│── military_dataset.yaml (original)
│── dataset.md

📊 Classes (12 categories)
ID	Class Name
0	camouflage_soldier
1	weapon
2	military_tank
3	military_truck
4	military_vehicle
5	civilian
6	soldier
7	civilian_vehicle
8	military_artillery
9	trench
10	military_aircraft
11	military_warship

⚙️ Setup Instructions
1. Install Dependencies

Run this inside Google Colab:

!pip install ultralytics -q

🚀 One-Cell Training Script (Used in This Project)
from pathlib import Path

# EDIT THIS → Path to your dataset folder in Google Drive
ROOT = Path("/content/drive/MyDrive/military_object_dataset")

# Create data.yaml automatically
open("/content/data.yaml", "w").write(f"""
train: {ROOT/'train/images'}
val: {ROOT/'val/images'}
test: {ROOT/'test/images'}
nc: 12
names: ['camouflage_soldier','weapon','military_tank','military_truck',
        'military_vehicle','civilian','soldier','civilian_vehicle',
        'military_artillery','trench','military_aircraft','military_warship']
""")

# Install YOLOv8
!pip install ultralytics -q
from ultralytics import YOLO

# Load model and train
model = YOLO("yolov8n.pt")  
model.train(
    data="/content/data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,
    name="exp_onecell"
)

🧪 Inference (Generate Predictions)
from ultralytics import YOLO

model = YOLO("/content/runs/detect/exp_onecell/weights/best.pt")

results = model.predict(
    source="/content/drive/MyDrive/military_object_dataset/test/images",
    save=True,
    save_txt=True,
    save_conf=True
)


Outputs will be saved here:

runs/detect/predict/labels/*.txt


Each .txt file contains:

class x_center y_center width height confidence

📈 Results
