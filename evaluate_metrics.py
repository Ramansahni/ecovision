import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from src.dataset import EcoVisionDataset

# Load model
model = load_model("eco_rgb_model.h5", compile=False)

# Create dataset
base_path = "data/processed/patches"

test_dataset = EcoVisionDataset(
    f"{base_path}/test/images",
    f"{base_path}/test/masks",
    batch_size=4,
    shuffle=False
)

# Predict
preds = model.predict(test_dataset, verbose=1)

# Collect masks
all_masks = []
for _, masks in test_dataset:
    all_masks.append(masks)
all_masks = np.concatenate(all_masks)

# Binarize
preds_binary = (preds > 0.5).astype(np.uint8)

all_preds = preds_binary.flatten()
all_masks = all_masks.flatten()

accuracy = np.mean(all_preds == all_masks)

intersection = np.sum(all_preds * all_masks)
union = np.sum(all_preds) + np.sum(all_masks) - intersection
iou = intersection / (union + 1e-7)

dice = (2 * intersection) / (np.sum(all_preds) + np.sum(all_masks) + 1e-7)

precision = precision_score(all_masks, all_preds)
recall = recall_score(all_masks, all_preds)
f1 = f1_score(all_masks, all_preds)

print("\n===== EcoVision Final Test Metrics =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Dice      : {dice:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

import matplotlib.pyplot as plt

# Take first batch
for images, masks in test_dataset:
    preds = model.predict(images)
    break

# Show first 3 samples
num_samples = 3

for i in range(num_samples):
    image = images[i]
    true_mask = masks[i].squeeze()
    pred_mask = (preds[i].squeeze() > 0.5).astype(float)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(true_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.show()