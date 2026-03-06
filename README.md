# Earth Observation: Delhi Airshed Land-Use Classification

**AI for Sustainability — SRIP 2026 | IIT Gandhinagar**

This project builds a complete land-use classification pipeline for the Delhi-NCR airshed region using multi-spectral satellite imagery from Sentinel-2 and ground-truth labels derived from the ESA WorldCover 2021 land cover product. The core workflow spans spatial data filtering, automated label construction from raster data, and supervised deep learning with a fine-tuned ResNet18 backbone — all implemented from scratch in a single reproducible Jupyter notebook.

---

## Motivation

Air quality monitoring over the Delhi National Capital Region requires a clear understanding of what types of land-use exist across the airshed — sprawling urban zones, agricultural tracts, vegetated corridors, water bodies, and bare/sparse terrain each have different emission and absorption profiles. Manually labelling thousands of satellite patches is impractical, so this project automates the process by cross-referencing high-resolution Sentinel-2 RGB imagery against the ESA WorldCover 10 m classification map and then training a convolutional neural network to predict land-use categories directly from the satellite images.

---

## Assignment Structure

| Question | Task | Marks |
|:---:|:---|:---:|
| Q1 | Spatial Reasoning & Data Filtering | 4 |
| Q2 | Label Construction & Dataset Preparation | 6 |
| Q3 | Model Training & Supervised Evaluation | 5 |
| | **Total** | **15** |

---

## Datasets

The project works with four complementary geospatial datasets, all sourced from a single [Kaggle dataset](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed):

| # | Dataset | Format | Description |
|---|---------|--------|-------------|
| 1 | **Delhi-NCR Region Boundary** | GeoJSON | Administrative shapefile covering 30 district-level polygons in the NCR |
| 2 | **Sentinel-2 RGB Patches** | PNG (128×128 px, 10 m/px) | 9,216 geo-referenced image patches covering the broader bounding box |
| 3 | **Delhi Airshed Boundary** | GeoJSON | Rectangular airshed polygon (~60 × 80 km) centred on Delhi |
| 4 | **ESA WorldCover 2021** | GeoTIFF | 10 m resolution land cover raster with 11 ESA-defined classes |

---

## Pipeline Overview

### Q1 — Spatial Reasoning & Data Filtering

1. Load the Delhi-NCR boundary shapefile and overlay a **60 × 60 km reference grid** to visualise coverage.
2. Parse all **9,216** Sentinel-2 patches (filename encodes `lat_lon`) and construct a GeoDataFrame.
3. Perform a **spatial join** against the NCR boundary to filter out patches that fall outside the region.
4. **8,015 patches** remain after filtering — these form the working dataset.

### Q2 — Label Construction & Dataset Preparation

1. For each filtered patch, read the corresponding **128 × 128 window** from the WorldCover raster.
2. Compute the **dominant (mode) land cover class** within the window, ignoring no-data pixels.
3. Map the 11 ESA classes into **5 simplified categories**:

| Simplified Category | ESA Source Classes |
|---|---|
| **Built-up** | Built-up (50) |
| **Cropland** | Cropland (40) |
| **Vegetation** | Tree cover (10), Shrubland (20), Grassland (30), Mangroves (95) |
| **Water** | Permanent water (80) |
| **Others** | Bare/sparse (60), Snow/ice (70), Herbaceous wetland (90), Moss/lichen (100) |

4. Final class distribution across the 8,015 labelled patches:

| Category | Count |
|---|---|
| Cropland | 5,473 |
| Built-up | 1,779 |
| Vegetation | 756 |
| Water | 7 |

5. Split into **train / validation / test** sets with stratified sampling (random seed = 42).

### Q3 — Model Training & Supervised Evaluation

- **Architecture:** ResNet18 with ImageNet-pretrained weights; final fully-connected layer replaced for 5-class output.
- **Training:** Cross-entropy loss, Adam optimiser, learning-rate scheduling, trained on GPU (CUDA).
- **Evaluation:** Accuracy, macro F1-score, per-class precision/recall, and a full confusion matrix on the held-out test set.
- **Outputs:** Training/validation loss & accuracy curves, confusion matrix heatmap, and saved `best_model.pth` checkpoint.

---

## Project Structure

```
earth-observation-delhi-airshed/
├── README.md
├── requirements.txt
├── .gitignore
├── earth_observation_notebook.ipynb   # Full pipeline (Q1 → Q2 → Q3)
├── data/
│   ├── delhi_ncr_region.geojson       # NCR boundary (30 districts)
│   ├── delhi_airshed.geojson          # Airshed boundary
│   ├── worldcover_bbox_delhi_ncr_2021.tif  # ESA WorldCover raster
│   └── rgb/                           # 9,216 Sentinel-2 image patches
│       ├── 28.2056_76.8558.png
│       └── ...
└── outputs/
    ├── best_model.pth                 # Trained ResNet18 weights
    ├── q1_grid_overlay.png            # NCR boundary with 60 km grid
    ├── q1_all_images_on_map.png       # All 9,216 patches plotted
    ├── q1_filtered_images.png         # Filtered patches (inside NCR)
    ├── q1_only_filtered.png           # Filtered patches only
    ├── q2_class_distribution.png      # Label distribution bar chart
    ├── q2_sample_images.png           # Sample images per class
    ├── q3_confusion_matrix.png        # Test set confusion matrix
    └── q3_training_curves.png         # Loss & accuracy over epochs
```

---

## Setup & Reproduction

### Prerequisites

- Python ≥ 3.10
- A CUDA-capable GPU is recommended for model training (the notebook auto-detects and uses GPU if available)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/earth-observation-delhi-airshed.git
cd earth-observation-delhi-airshed
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `matplotlib`, `geopandas`, `shapely`, `rasterio`, `Pillow`, `scikit-learn`, `torch`, `torchvision`, `seaborn`, `tqdm`.

### 3. Download the data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed) and place the files in the `data/` directory. The directory should contain:
- `delhi_ncr_region.geojson`
- `delhi_airshed.geojson`
- `worldcover_bbox_delhi_ncr_2021.tif`
- `rgb/` folder with all image patches

### 4. Run the notebook

```bash
jupyter notebook earth_observation_notebook.ipynb
```

Execute cells sequentially — the notebook is structured in order (Q1 → Q2 → Q3) and each section builds on the outputs of the previous one.

---

## Key Results

- **9,216** total Sentinel-2 patches loaded; **8,015** retained after NCR spatial filtering.
- **8,015** patches successfully labelled using ESA WorldCover majority-vote mapping.
- **ResNet18** (pretrained on ImageNet, fine-tuned) achieves strong classification performance across the 5 land-use categories.
- Full evaluation metrics (accuracy, macro F1-score, per-class precision/recall) and confusion matrix are generated in the notebook.
- Training and validation curves are saved to `outputs/q3_training_curves.png`.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| PyTorch + Torchvision | Deep learning (ResNet18 transfer learning) |
| GeoPandas + Shapely | Geospatial data handling and spatial joins |
| Rasterio | Reading GeoTIFF raster data (WorldCover) |
| Matplotlib + Seaborn | Static visualisations and plots |
| scikit-learn | Train/test splitting, classification metrics |

---

## Author

**Kush Patel**
M.Tech-ICT (Machine Learning Specialisation) | Dhirubhai Ambani University
SRIP 2026 Intern — IIT Gandhinagar
