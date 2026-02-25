import os
import numpy as np
import tensorflow as tf


class EcoVisionDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=4, shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        # ----------------------------
        # FIX: Proper image-mask pairing
        # ----------------------------
        image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".npy")
        ])

        mask_files = set([
            f for f in os.listdir(mask_dir) if f.endswith(".npy")
        ])

        paired_images = []

        for img_name in image_files:
            number = img_name.replace("img_", "")
            mask_name = "mask_" + number

            if mask_name in mask_files:
                paired_images.append(img_name)

        print(f"Total paired samples: {len(paired_images)}")

        self.image_files = paired_images
        self.indices = np.arange(len(self.image_files))

        self.on_epoch_end()

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        images = []
        masks = []

        for i in batch_indices:
            image_name = self.image_files[i]

            img_path = os.path.join(self.image_dir, image_name)

            # Match mask name properly
            number = image_name.replace("img_", "")
            mask_name = "mask_" + number
            mask_path = os.path.join(self.mask_dir, mask_name)

            # ----------------------------
            # LOAD IMAGE + MASK
            # ----------------------------
            image = np.load(img_path)      # (4, 256, 256)
            mask = np.load(mask_path)      # (256, 256)

            # ----------------------------
            # FIX CHANNEL ORDER
            # ----------------------------
            # Convert (C, H, W) → (H, W, C)
            image = np.transpose(image, (1, 2, 0))   # (256, 256, 4)

            # Remove B8 (keep first 3 channels)
            image = image[:, :, :3]                  # (256, 256, 3)

            # ----------------------------
            # TYPE CONVERSION
            # ----------------------------
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

            # ----------------------------
            # NORMALIZATION (SAFE)
            # ----------------------------
            max_val = np.max(image)
            if max_val > 0:
                image = image / max_val

            # ----------------------------
            # FIX MASK SHAPE
            # ----------------------------
            mask = np.expand_dims(mask, axis=-1)     # (256, 256, 1)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)