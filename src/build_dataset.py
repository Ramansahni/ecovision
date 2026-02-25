import os
import numpy as np
import rasterio
from tqdm import tqdm
import random
import shutil

# ================= SETTINGS =================

RAW_BASE = "data/raw/gee_regions"
OUTPUT_BASE = "data/processed/patches"

PATCH_SIZE = 256
STRIDE = 128
FOREST_THRESHOLD = 0.4

TARGETS = {
    "western_ghats": 300,
    "himachal": 150
}

# =========================================================

def ensure_clean_output():
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)

    os.makedirs(os.path.join(OUTPUT_BASE, "train/images"))
    os.makedirs(os.path.join(OUTPUT_BASE, "train/masks"))
    os.makedirs(os.path.join(OUTPUT_BASE, "test/images"))
    os.makedirs(os.path.join(OUTPUT_BASE, "test/masks"))


def process_region(region_name):

    print("\n" + "="*70)
    print(f"PROCESSING REGION: {region_name}")
    print("="*70)

    region_path = os.path.join(RAW_BASE, region_name)
    tif_file = [f for f in os.listdir(region_path) if f.endswith(".tif")][0]
    tif_path = os.path.join(region_path, tif_file)

    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)

    # Replace NaN
    data = np.nan_to_num(data, nan=0.0)

    height = data.shape[1]
    width = data.shape[2]

    print(f"Image size: {height} x {width}")

    red = data[2]
    nir = data[3]

    ndvi = (nir - red) / (nir + red + 1e-6)

    forest_mask = ndvi > FOREST_THRESHOLD

    print("NDVI stats:")
    print(f"  Min: {ndvi.min():.3f}")
    print(f"  Max: {ndvi.max():.3f}")
    print(f"  Mean: {ndvi.mean():.3f}")

    patches = []

    total_scanned = 0
    rejected = 0

    for y in range(0, height - PATCH_SIZE, STRIDE):
        for x in range(0, width - PATCH_SIZE, STRIDE):

            total_scanned += 1

            patch = data[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = forest_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # Filter invalid patches
            valid_pixels = np.count_nonzero(patch)
            valid_ratio = valid_pixels / patch.size

            max_reflectance = patch.max()
            forest_ratio = mask_patch.mean()

            if valid_ratio < 0.95:
                rejected += 1
                continue

            if max_reflectance < 500:
                rejected += 1
                continue

            if forest_ratio < 0.05 or forest_ratio > 0.95:
                rejected += 1
                continue

            balance_score = abs(forest_ratio - 0.5)

            patches.append({
                "patch": patch,
                "mask": mask_patch.astype(np.float32),
                "score": balance_score
            })

    print(f"Total patches scanned: {total_scanned}")
    print(f"Rejected patches: {rejected}")
    print(f"Valid patches collected: {len(patches)}")

    # Sort by balance
    patches.sort(key=lambda x: x["score"])

    target = TARGETS[region_name]
    selected = patches[:target]

    print(f"Selected best {len(selected)} patches.")

    return selected


def save_dataset(all_patches):

    random.shuffle(all_patches)

    split_index = int(len(all_patches) * 0.8)

    train_set = all_patches[:split_index]
    test_set = all_patches[split_index:]

    print(f"\nFinal dataset size: {len(all_patches)}")
    print(f"Train: {len(train_set)}")
    print(f"Test: {len(test_set)}")

    def save_split(split_data, split_name):

        for i, item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):

            img = item["patch"] / 10000.0  # scale reflectance
            mask = item["mask"]

            img_path = os.path.join(OUTPUT_BASE, f"{split_name}/images/img_{i:04d}.npy")
            mask_path = os.path.join(OUTPUT_BASE, f"{split_name}/masks/mask_{i:04d}.npy")

            np.save(img_path, img.astype(np.float32))
            np.save(mask_path, mask)

    save_split(train_set, "train")
    save_split(test_set, "test")


# ================= MAIN =================

if __name__ == "__main__":

    ensure_clean_output()

    all_patches = []

    for region in TARGETS.keys():
        region_patches = process_region(region)
        all_patches.extend(region_patches)

    save_dataset(all_patches)

    print("\nDataset building complete.")