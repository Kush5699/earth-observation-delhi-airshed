# 🌍 Earth Observation: Delhi Airshed Land-Use Classification

**AI for Sustainability — SRIP 2026 | IIT Gandhinagar**

An AI-based audit pipeline for the Delhi Airshed to identify land-use patterns using Sentinel-2 satellite imagery and ESA WorldCover 2021 data.

## 📋 Task Overview

| Question | Task | Marks |
|:---:|:---|:---:|
| Q1 | Spatial Reasoning & Data Filtering | 4 |
| Q2 | Label Construction & Dataset Preparation | 6 |
| Q3 | Model Training & Supervised Evaluation | 5 |
| | **Total** | **15** |

## 🗂️ Project Structure

```
earth-observation-delhi-airshed/
├── README.md
├── requirements.txt
├── earth_observation.py          # Complete solution script
├── data/                         # Place datasets here
│   ├── delhi_ncr_region.geojson
│   ├── delhi_airshed.geojson
│   ├── worldcover_bbox_delhi_ncr_2021.tif
│   └── rgb/                      # Sentinel-2 image patches
│       ├── 28.3456_77.1234.png
│       └── ...
└── outputs/                      # Generated outputs
    ├── q1_grid_overlay.png
    ├── q2_class_distribution.png
    ├── q3_confusion_matrix.png
    └── q3_training_curves.png
```

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed) and place files in the `data/` directory.

### 3. Run

```bash
python earth_observation.py
```

Or open the Jupyter notebook version for step-by-step execution.

## 📊 Results Summary

- **Images before filtering:** Reported in output
- **Images after filtering:** Reported in output
- **Model:** ResNet18 (pretrained, fine-tuned)
- **Metrics:** Accuracy, F1-score (macro), Confusion Matrix

## 📦 Dataset Sources

- [Delhi NCR Region Shapefile](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_ncr_region.geojson)
- [Sentinel-2 RGB Patches](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=rgb)
- [Delhi Airshed Shapefile](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_airshed.geojson)
- [ESA WorldCover 2021](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=worldcover_bbox_delhi_ncr_2021.tif)

## 👤 Author

**Kush Patel**
M.Tech-ICT (Machine Learning Specialization) | Dhirubhai Ambani University
