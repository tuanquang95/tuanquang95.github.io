---
title: "AICITY Helmet Detection"
excerpt: "Helmet Detection - AICITY Challenge"
collection: portfolio
---

# 🪖 Helmet Violation Detection — NVIDIA AI City Challenge 2023 · Track 5

[![Challenge](https://img.shields.io/badge/NVIDIA%20AI%20City%20Challenge-2023%20Track%205-76b900?style=flat-square&logo=nvidia)](https://www.aicitychallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-YOLOv8-FF4500?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-tuanquang95%2Faio__pending__track5-181717?style=flat-square&logo=github)](https://github.com/tuanquang95/aio_pending_track5)

> **Detecting motorcycle helmet rule violations in real-time surveillance video** — a 7-class object detection system built for the CVPR 20
---

## 📌 Overview

Motorcycle accidents are a leading cause of traffic fatalities, especially in developing countries where helmet compliance is inconsistently enforced. This project addresses **Track 5** of the [NVIDIA AI City Challenge 2023](https://www.aicitychallenge.org/), which tasks participants with automatically detecting helmet violations for motorcyclists in surveillance footage.

The system identifies up to **7 object classes per frame** — the motorcycle, driver, and up to two passengers — and classifies each rider as wearing or not wearing a helmet.

---

## 🏁 Challenge Details

| Property | Value |
|---|---|
| **Challenge** | NVIDIA AI City Challenge 2023 |
| **Track** | Track 5 — Detecting Violation of Helmet Rule for Motorcyclists |
| **Workshop** | CVPR 2023 |
| **Training Set** | 100 videos × 20 seconds @ 10 FPS, 1920×1080 |
| **Test Set** | 100 videos (same format, labels withheld) |
| **Evaluation Metric** | Mean Average Precision (mAP) |

### 🏷️ Detection Classes

| Class ID | Label | Description |
|---|---|---|
| 0 | `motorbike` | The motorcycle itself |
| 1 | `DHelmet` | Driver — wearing helmet ✅ |
| 2 | `DNoHelmet` | Driver — no helmet ❌ |
| 3 | `P1Helmet` | Passenger 1 — wearing helmet ✅ |
| 4 | `P1NoHelmet` | Passenger 1 — no helmet ❌ |
| 5 | `P2Helmet` | Passenger 2 — wearing helmet ✅ |
| 6 | `P2NoHelmet` | Passenger 2 — no helmet ❌ |

---

## 🧠 Approach

### Detection Pipeline

```
Raw Video Frames
      │
      ▼
 Frame Extraction (10 FPS)
      │
      ▼
 Object Detection (YOLOv8 / custom backbone)
      │
      ├──► Motorcycle BBoxes
      └──► Rider BBoxes + Helmet Classification
                │
                ▼
        Post-Processing (NMS, score thresholding)
                │
                ▼
        Submission File (.txt per video)
```

### Key Techniques

- **Multi-scale training** — trained at multiple input resolutions to handle both distant and close-up riders
- **Data augmentation** — mosaic, mixup, random flip, HSV shifts, and copy-paste augmentation to improve generalization across varied lighting and occlusion conditions
- **Pseudo-labeling** — generated soft labels on unlabeled frames to expand effective training data
- **Test-Time Augmentation (TTA)** — horizontal flip and multi-scale inference averaged at prediction time
- **Weighted Box Fusion (WBF)** — ensemble-level bounding box merging across multiple model checkpoints to improve precision

---

## 📁 Repository Structure

```
aio_pending_track5/
├── data/
│   ├── raw/                    # Original dataset videos
│   ├── frames/                 # Extracted frames (10 FPS)
│   ├── annotations/            # YOLO-format labels
│   └── splits/                 # train / val split configs
├── configs/
│   ├── model.yaml              # Model architecture config
│   └── hyp.yaml                # Hyperparameter config
├── src/
│   ├── extract_frames.py       # Video → frame extraction
│   ├── train.py                # Training entry point
│   ├── detect.py               # Inference on test videos
│   ├── postprocess.py          # NMS, WBF, score filtering
│   └── utils/
│       ├── augmentation.py     # Custom augmentation helpers
│       └── submission.py       # Format output for AIC submission
├── notebooks/
│   ├── EDA.ipynb               # Dataset exploration
│   └── Evaluation.ipynb        # mAP analysis and error inspection
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Install Dependencies

```bash
git clone https://github.com/tuanquang95/aio_pending_track5.git
cd aio_pending_track5
pip install -r requirements.txt
```

### 2. Prepare Data

Download the Track 5 dataset from the [AI City Challenge portal](https://www.aicitychallenge.org/) and extract it:

```bash
# Extract video frames at 10 FPS
python src/extract_frames.py \
    --video-dir data/raw/videos/ \
    --output-dir data/frames/ \
    --fps 10
```

### 3. Train

```bash
python src/train.py \
    --data configs/model.yaml \
    --img-size 1280 \
    --batch-size 16 \
    --epochs 100 \
    --weights yolov8x.pt \
    --device 0
```

### 4. Inference & Generate Submission

```bash
python src/detect.py \
    --source data/raw/test_videos/ \
    --weights runs/train/exp/weights/best.pt \
    --conf 0.25 \
    --iou-thres 0.45 \
    --img-size 1280 \
    --output submissions/result.txt
```

---

## 📊 Results

| Model | Input Size | mAP@0.5 | Notes |
|---|---|---|---|
| YOLOv8m baseline | 640 | — | Initial benchmark |
| YOLOv8x | 1280 | — | Full-res training |
| YOLOv8x + TTA | 1280 | — | With test-time augmentation |
| **YOLOv8x + TTA + WBF** | **1280** | **Best** | Final submission |

> *Exact mAP scores on the official test leaderboard are tied to the challenge evaluation server. Validation set metrics are tracked in `notebooks/Evaluation.ipynb`.*

---

## 🔍 Challenges & Observations

- **Class imbalance** — `DHelmet` instances vastly outnumber `P2NoHelmet`, requiring careful sampling strategies during training
- **Small object detection** — distant motorcycles with tiny rider bounding boxes were difficult to classify reliably; high-resolution inputs (1280+) were critical
- **Occlusion** — riders stacked on motorcycles heavily overlap each other, making passenger 1 and 2 classification particularly challenging
- **Lighting variance** — the dataset spans daytime, nighttime, and mixed-lighting scenes, requiring strong HSV augmentation

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Detection Framework | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Training Accelerator | CUDA / GPU (single / multi-GPU) |
| Data Processing | OpenCV, FFmpeg |
| Experiment Tracking | Weights & Biases |
| Post-processing | Ensemble Boxes (WBF) |
| Notebook Environment | Jupyter / Google Colab |

---

## 📚 References

- [NVIDIA AI City Challenge 2023 — Official Site](https://www.aicitychallenge.org/)
- [Track 5 Data & Evaluation Details](https://www.aicitychallenge.org/2023-data-and-evaluation/)
- Ultralytics YOLOv8 — [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Solovyev et al., *Weighted Boxes Fusion*, 2021 — ensemble box merging for object detection
- Tsai et al., *Video Analytics for Detecting Motorcyclist Helmet Rule Violations*, CVPRW 2023

---

*Built for the NVIDIA AI City Challenge 2023 CVPR Workshop · Track 5*
