import os
import numpy as np
import rasterio

RAW_BASE = "data/raw/gee_regions"

def inspect_region(region_name):
    print("\n" + "="*60)
    print(f"INSPECTING REGION: {region_name}")
    print("="*60)

    region_path = os.path.join(RAW_BASE, region_name)
    tif_files = [f for f in os.listdir(region_path) if f.endswith(".tif")]

    if not tif_files:
        print("No TIFF found.")
        return

    tif_path = os.path.join(region_path, tif_files[0])
    print(f"File: {tif_files[0]}")

    with rasterio.open(tif_path) as src:
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Data type: {src.dtypes}")

        data = src.read()  # (bands, H, W)

        print("\nBand-wise stats:")
        for i in range(data.shape[0]):
            band = data[i]
            print(f" Band {i+1}: min={np.nanmin(band):.2f}, max={np.nanmax(band):.2f}, mean={np.nanmean(band):.2f}")

        nan_pixels = np.isnan(data).sum()
        print(f"\nTotal NaN pixels: {nan_pixels}")

inspect_region("western_ghats")
inspect_region("himachal")