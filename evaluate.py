import tensorflow as tf
from src.losses import bce_dice_loss, iou_metric
from src.dataset import EcoVisionDataset

model = tf.keras.models.load_model(
    "eco_rgb_model.h5",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "iou_metric": iou_metric
    }
)

test_dataset = EcoVisionDataset(
    "data/processed/patches/test/images",
    "data/processed/patches/test/masks",
    batch_size=4,
    shuffle=False
)

results = model.evaluate(test_dataset)

print("Test Loss:", results[0])
print("Test Accuracy:", results[1])
print("Test IoU:", results[2])