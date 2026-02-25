import tensorflow as tf
from src.model import build_unet
from src.dataset import EcoVisionDataset


# =============================
# LOSS + METRICS (inside file)
# =============================

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return intersection / (union + 1e-6)


# =============================
# DATASETS
# =============================

train_dataset = EcoVisionDataset(
    "data/processed/patches/train/images",
    "data/processed/patches/train/masks",
    batch_size=4
)

val_dataset = EcoVisionDataset(
    "data/processed/patches/test/images",
    "data/processed/patches/test/masks",
    batch_size=4
)


# =============================
# MODEL
# =============================

model = build_unet((256, 256, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=["accuracy", iou_metric]
)

model.summary()

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25
)

model.save("eco_rgb_model.h5")