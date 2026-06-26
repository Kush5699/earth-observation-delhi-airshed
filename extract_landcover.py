"""
Extract real land cover data from worldcover_bbox_delhi_ncr_2021.tif
and export as a JSON grid for the interactive map visualization.

ESA WorldCover 2021 Class Codes:
  10 = Tree cover
  20 = Shrubland
  30 = Grassland
  40 = Cropland
  50 = Built-up
  60 = Bare / sparse vegetation
  80 = Permanent water bodies
  90 = Herbaceous wetland
  95 = Mangroves
  100 = Moss and lichen
"""

import rasterio
import numpy as np
import json
from collections import Counter

# --- Config ---
TIF_PATH = r"data\worldcover_bbox_delhi_ncr_2021.tif"
OUTPUT_PATH = r"data\landcover_grid.json"
GRID_STEP = 0.005  # ~500m resolution for the grid (degrees)

# ESA WorldCover class definitions
CLASS_MAP = {
    10:  {"label": "Tree Cover",          "color": "#27ae60"},
    20:  {"label": "Shrubland",           "color": "#1a5e35"},
    30:  {"label": "Grassland",           "color": "#8b9467"},
    40:  {"label": "Cropland",            "color": "#f39c12"},
    50:  {"label": "Built-up",            "color": "#e74c3c"},
    60:  {"label": "Bare / Sparse",       "color": "#95a5a6"},
    80:  {"label": "Water Bodies",        "color": "#3498db"},
    90:  {"label": "Herbaceous Wetland",  "color": "#1abc9c"},
    95:  {"label": "Mangroves",           "color": "#16a085"},
    100: {"label": "Moss / Lichen",       "color": "#bdc3c7"},
}

def main():
    print("=" * 60)
    print("🗺️  Land Cover Extraction Script")
    print("=" * 60)

    # 1. Read the TIF
    print(f"\n📂 Reading TIF: {TIF_PATH}")
    with rasterio.open(TIF_PATH) as src:
        data = src.read(1)  # Band 1
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

        print(f"   Shape: {data.shape}")
        print(f"   CRS: {crs}")
        print(f"   Bounds: W={bounds.left:.4f}, E={bounds.right:.4f}, "
              f"S={bounds.bottom:.4f}, N={bounds.top:.4f}")

        # Show class distribution
        unique, counts = np.unique(data, return_counts=True)
        total_pixels = data.size
        print(f"\n📊 Class Distribution ({total_pixels:,} total pixels):")
        for cls, cnt in zip(unique, counts):
            pct = cnt / total_pixels * 100
            label = CLASS_MAP.get(cls, {}).get("label", f"Unknown ({cls})")
            print(f"   Class {cls:3d} ({label:25s}): {cnt:>10,} pixels ({pct:.1f}%)")

        # 2. Create grid
        lat_min, lat_max = bounds.bottom, bounds.top
        lon_min, lon_max = bounds.left, bounds.right

        print(f"\n🔲 Generating grid (step={GRID_STEP}°, ~{GRID_STEP * 111:.0f}km)...")

        grid_cells = []
        lat = lat_min
        while lat < lat_max:
            lon = lon_min
            while lon < lon_max:
                # Get pixel coordinates for this grid cell
                # Top-left corner
                col_start, row_start = ~transform * (lon, lat + GRID_STEP)
                # Bottom-right corner
                col_end, row_end = ~transform * (lon + GRID_STEP, lat)

                # Clamp to valid ranges
                row_start = max(0, int(row_start))
                row_end = min(data.shape[0], int(row_end))
                col_start = max(0, int(col_start))
                col_end = min(data.shape[1], int(col_end))

                if row_start < row_end and col_start < col_end:
                    # Get the most common class in this cell (majority vote)
                    cell_data = data[row_start:row_end, col_start:col_end]
                    if cell_data.size > 0:
                        # Use majority class
                        values, value_counts = np.unique(cell_data, return_counts=True)
                        majority_class = int(values[np.argmax(value_counts)])

                        if majority_class in CLASS_MAP:
                            grid_cells.append({
                                "lat": round(lat, 5),
                                "lon": round(lon, 5),
                                "cls": majority_class,
                            })

                lon += GRID_STEP
            lat += GRID_STEP

        print(f"   Generated {len(grid_cells):,} grid cells")

        # 3. Count cells per class
        class_counts = Counter(c["cls"] for c in grid_cells)
        print(f"\n📊 Grid Cell Distribution:")
        for cls, cnt in sorted(class_counts.items()):
            label = CLASS_MAP.get(cls, {}).get("label", "?")
            print(f"   {label:25s}: {cnt:>5} cells")

        # 4. Build output JSON
        output = {
            "step": GRID_STEP,
            "bounds": {
                "lat_min": round(lat_min, 5),
                "lat_max": round(lat_max, 5),
                "lon_min": round(lon_min, 5),
                "lon_max": round(lon_max, 5),
            },
            "classes": CLASS_MAP,
            "cells": grid_cells,
        }

        # 5. Save
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f)

        file_size_mb = len(json.dumps(output)) / (1024 * 1024)
        print(f"\n✅ Saved to: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
        print(f"   Total cells: {len(grid_cells):,}")
        print(f"   Grid step: {GRID_STEP}° (~{GRID_STEP * 111:.0f}m)")
        print("=" * 60)


if __name__ == "__main__":
    main()
