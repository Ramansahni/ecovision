import os
import numpy as np
import matplotlib.pyplot as plt
import random

BASE = "data/processed/patches/train"

images = os.listdir(os.path.join(BASE, "images"))

print(f"Total training samples: {len(images)}")

samples = random.sample(images, 5)

for img_name in samples:

    img_path = os.path.join(BASE, "images", img_name)
    mask_path = os.path.join(BASE, "masks", img_name.replace("img_", "mask_"))

    img = np.load(img_path)
    mask = np.load(mask_path)

    red = img[2]
    nir = img[3]

    ndvi = (nir - red) / (nir + red + 1e-6)

    forest_ratio = mask.mean()

    print("\n" + "-"*50)
    print(f"Image: {img_name}")
    print(f"Forest ratio: {forest_ratio:.3f}")
    print(f"Min reflectance: {img.min():.3f}")
    print(f"Max reflectance: {img.max():.3f}")

    rgb = np.transpose(img[[2,1,0]], (1,2,0))
    rgb = np.clip(rgb * 3, 0, 1)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(ndvi, cmap="RdYlGn")
    plt.title("NDVI")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.show()