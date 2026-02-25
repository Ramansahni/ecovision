import os
import numpy as np
import rasterio

RAW_BASE = "data/raw/gee_regions"

def analyze_ndvi(region_name):
    print("\n" + "="*60)
    print(f"NDVI ANALYSIS: {region_name}")
    print("="*60)

    region_path = os.path.join(RAW_BASE, region_name)
    tif_file = [f for f in os.listdir(region_path) if f.endswith(".tif")][0]
    tif_path = os.path.join(region_path, tif_file)

    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)

    # Replace NaN with 0
    data = np.nan_to_num(data, nan=0.0)

    blue = data[0]
    green = data[1]
    red = data[2]
    nir = data[3]

    ndvi = (nir - red) / (nir + red + 1e-6)

    print(f"NDVI min: {ndvi.min():.3f}")
    print(f"NDVI max: {ndvi.max():.3f}")
    print(f"NDVI mean: {ndvi.mean():.3f}")

    forest_area = np.sum(ndvi > 0.4)
    total_area = ndvi.size

    print(f"Pixels with NDVI > 0.4: {forest_area}")
    print(f"Total pixels: {total_area}")
    print(f"Forest percentage: {(forest_area / total_area) * 100:.2f}%")

analyze_ndvi("western_ghats")
analyze_ndvi("himachal")