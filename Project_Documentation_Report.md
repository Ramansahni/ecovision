# EcoVision: Comprehensive Project Documentation & Analysis Report

## 1. Executive Summary & Project Overview

The **EcoVision** project is an advanced Artificial Intelligence and Remote Sensing initiative designed to accomplish two primary objectives:
1. **Automated Forest Segmentation**: Using deep learning to accurately identify and segment forested areas from high-resolution satellite imagery.
2. **Eco-Friendly Path Planning**: Utilizing geolocated environmental cost maps to compute optimal, low-impact routes through natural landscapes.

By integrating state-of-the-art Computer Vision algorithms with classical pathfinding heuristics, EcoVision bridges the gap between raw Earth Observation (EO) data and actionable ecological intelligence. This report provides an exhaustive, multi-page level description of the entire project structure, the data engineering pipeline, the neural network architecture, the training modalities, performance evaluation, and the robust routing engine.

---

## 2. Directory and Workspace Architecture

The workspace is organized into distinct logical modules, segregating raw data, processed datasets, source code, execution scripts, and Earth Observation bands.

### 2.1 The Root Directory (`/EcoVision`)
The root directory serves as the execution epicenter. It contains:
- **`README.md` & `requirements.txt`**: Standard project definition files.
- **Execution Scripts (`train.py`, `evaluate.py`, `evaluate_local.py`, `evaluate_metrics.py`)**: Provide end-to-end functionality for deep learning model training, loading, and evaluation.
- **`eco_path_planner.py`**: A standalone application for calculating eco-optimal routes based on vegetation density.
- **Model Artifacts (`eco_rgb_model.h5`, `ecovision_model.keras`, `ecovision_final.keras`)**: The serialized weights and graph definitions of the trained U-Net architectures.

### 2.2 The Source Directory (`/src`)
This folder encapsulates the core modular logic of the machine learning pipeline:
- **`build_dataset.py`**: Automates the extraction of usable image patches from heavy TIFFs.
- **`dataset.py`**: Defines the `tf.keras.utils.Sequence` data generator.
- **`inspect_raw_data.py` & `ndvi_analysis.py`**: Analytical scripts to probe the raw Google Earth Engine (GEE) imagery.
- **`visualize_processed.py`**: For sanity-checking patches before feeding them to the neural network.
- **`model.py`**: Contains the Convolutional Neural Network (CNN) architecture (U-Net).
- **`metrics.py`**: Implements custom loss functions (Focal Loss) and evaluation metrics (IoU).

### 2.3 The Data Directory (`/data`)
Separated into two main stages:
- **`/data/raw/gee_regions`**: Contains original, massive multi-band `.tif` files downloaded indicating distinct geographical zones (e.g., `western_ghats`, `himachal`).
- **`/data/processed/patches`**: Contains meticulously extracted 256x256 `.npy` arrays, strictly partitioned into `/train` and `/test` folders, further separated into `/images` and `/masks`.

### 2.4 The Earth Observation Directory (`/EO`)
This directory holds sample remote sensing bands for the path-planning module:
- **Bands**: `b2.tiff` (Blue), `b3.tiff` (Green), `b4.tiff` (Red), and `b8.tiff` (Near-Infrared / NIR).

---

## 3. Data Engineering & Processing Pipeline

The transformation from massive satellite imagery to model-ready tensors is handled comprehensively by the project's data engineering suite.

### 3.1 Raw Data Ingestion & Inspection
The pipeline begins with `src/inspect_raw_data.py`. This script leverages the `rasterio` spatial data library to read the massive `.tif` files from `data/raw/gee_regions`. It outputs vital metadata including the coordinate reference system (CRS), spatial resolution (Width x Height), band configurations, and detects structural anomalies like `NaN` pixels. 

### 3.2 Spectral Analysis (NDVI Calculation)
The `src/ndvi_analysis.py` script is tasked with computing the **Normalized Difference Vegetation Index (NDVI)**. NDVI is a standard indicator of live green vegetation, defined mathematically as:
`NDVI = (NIR - Red) / (NIR + Red)`
The script extracts the Red (Band 3) and NIR (Band 4) channels from the raw arrays. It establishes a binary threshold (e.g., NDVI > 0.4) to classify a pixel as "Forest". It outputs the total forest area and percentage, giving a macro-view of the region's ecological density.

### 3.3 Patch Extraction and Dataset formulation
The heavy lifting is performed by `src/build_dataset.py`. Because satellite images are too massive to fit into GPU memory, they must be tiled.
1. **Tiling Strategy**: The script uses a sliding window approach to extract `256x256` pixel patches with a defined `STRIDE` (128 pixels, allowing for 50% overlap).
2. **Quality Control**: Patches are fiercely vetted. If a patch consists of >5% invalid (NaN/0) pixels, or if its maximum reflectance is too low (indicating shadow/clouds), it is rejected.
3. **Class Balancing**: Patches where the forest ratio is extremely low (<5%) or exceptionally high (>95%) are discarded to prevent the model from becoming biased toward entirely empty or completely saturated masks.
4. **Targeted Sampling**: To prevent over-representation, a specific target number of patches is collected per region (`western_ghats`: 300, `himachal`: 150).
5. **Serialization**: The approved patches and their corresponding binary NDVI masks are saved as `.npy` format into train (80%) and test (20%) directories. Visualizations of these processed arrays can be invoked using `src/visualize_processed.py`.

### 3.4 Keras Dataset Generator
To feed data efficiently during training, `src/dataset.py` defines `EcoVisionDataset`, inheriting from `tf.keras.utils.Sequence`.
- **File Alignment**: It mathematically ensures that `img_X.npy` strictly matches `mask_X.npy`.
- **Channel Modification**: Though patches might possess 4 bands initially, the generator strips out the NIR band, transforming the tensor from `(4, 256, 256)` to `(256, 256, 3)` (standard RGB).
- **Normalization**: Pixel combinations are safely normalized to a `[0, 1]` float32 range.
- **Batching & Shuffling**: Data is batched (batch_size=4) and aggressively shuffled at the end of every epoch to prevent gradient oscillation.

---

## 4. Deep Learning Architecture: The U-Net Model

The segmentation engine is defined in `src/model.py`. The project utilizes a **U-Net** topology, renowned for biomedical and satellite image segmentation due to its ability to retain high-frequency spatial details.

### 4.1 Encoder (Contraction Path)
The encoder extracts deep semantic features by progressively down-sampling the input RGB image.
- **Blocks**: It consists of three primary convolutional blocks. Each block utilizes two stacked `Conv2D` layers (3x3 kernel, "same" padding), immediately followed by `BatchNormalization` (to stabilize learning) and a `ReLU` non-linear activation.
- **Pooling**: Following each block, a `MaxPooling2D` operation reduces the spatial dimensions by a factor of 2, while the channel depth increases (16 → 32 → 64).

### 4.2 Bottleneck
The absolute nadir of the network bridges the encoder and decoder. It employs a high-capacity convolutional block with 128 filters, capturing the densest, most abstract semantic representation of the image.

### 4.3 Decoder (Expansion Path)
The decoder reconstructs the spatial dimensions to output a full-resolution segmentation mask.
- **Up-Sampling**: Symmetrical `UpSampling2D` layers increase the spatial resolution.
- **Skip Connections**: Crucially, the up-sampled feature maps are concatenated with the temporally corresponding feature maps from the encoder (e.g., `Concatenate()([u5, c3])`). This allows the decoder to recover fine-grained, localized spatial details lost during max-pooling.
- **Output Layer**: A final 1x1 convolution collapses the feature channels down to a single dimension (depth=1). A `sigmoid` activation restricts the output between `0` and `1`, representing the raw probability of a pixel belonging to the "Forest" class.

---

## 5. Training Methodology & Execution

Model training is orchestrated inside the root `train.py` script.

### 5.1 Optimization & Loss Functions
The model uses the `Adam` optimizer with a conservative learning rate of `1e-4`. 
The loss landscape in semantic segmentation is often plagued by "class imbalance" (e.g., significantly more non-forest background than actual forest). To counter this, a custom hybrid loss function is used:
- **BCE + Dice Loss**: `bce_dice_loss`. It combines standard `Binary Cross-Entropy` (excellent for pixel-wise probability classification) with `Dice Loss`.
- **Dice Loss**: A differentiable counterpart to the F1 score. It calculates `1 - (2 * Intersection / Union)`. It forces the model to prioritize spatial overlap over sheer background accuracy.
*(Note: `src/metrics.py` also defines a `Focal Loss` function to heavily penalize hard-to-classify examples, showing the evolutionary attempts at optimizing model robustness).*

### 5.2 Training Lifecycle
The model accepts the `train_dataset` sequence and validates against the `val_dataset`. It is trained over 25 epochs. Upon culmination, the model weights and architecture are serialized and saved natively as `eco_rgb_model.h5`.

---

## 6. Evaluation Protocols

Once trained, the model is strictly peer-reviewed using evaluation scripts located in the root directory.

### 6.1 Inference & Basic Evaluation
`evaluate_local.py` is a lightweight script to verify that the `.h5` model loads into memory correctly mapping custom objects (`bce_dice_loss`, `iou_metric`). `evaluate.py` handles the primary `model.evaluate()` Keras loops, logging the overall Test Loss, Accuracy, and Intersection-over-Union (IoU).

### 6.2 Advanced Granular Metrics
`evaluate_metrics.py` provides a clinical breakdown of the model's predictive power. It pushes the entire test set through the model, flattens the output masks, and cross-references them against ground truths to calculate:
- **Accuracy**: Raw pixel matching.
- **IoU (Jaccard Index)**: The golden standard for segmentation.
- **Dice Coefficient**: Strict overlap evaluation.
- **Precision**: Confidence of the model when it decrees a pixel as "Forest".
- **Recall**: Proportion of the actual forest the model successfully captured.
- **F1 Score**: Harmonic mean of Precision and Recall.

Furthermore, it employs `matplotlib` to render side-by-side visual comparisons: Input Image vs. True Mask vs. Predicted Mask.

---

## 7. The Eco-Friendly Path Planner

Beyond deep learning, the project features `eco_path_planner.py`, a robust heuristic routing algorithm that finds low-impact nature routes. 

### 7.1 Environmental Heuristics & Cost Mapping
1. **Band Fusion**: It loads raw `b2`, `b3`, `b4`, and `b8` TIFFs from the `EO` folder.
2. **NDVI to FVC**: It calculates NDVI and transforms it into **Fractional Vegetation Cover (FVC)** using empirical normalizations. FVC estimates the direct percentage of the ground covered by vegetation shadow.
3. **Topological Cost Map**: A mathematical grid is constructed where moving through a pixel has a strict computational "cost". The formula `1 + (fvc * 10)` means dense vegetation heavily penalizes the route (to encourage preservation and prevent disruption). Water bodies (NDVI < 0) are assigned extreme absolute penalties (cost=100) acting as complete obstacles.

### 7.2 The A* (A-Star) Algorithm
To traverse this complex cost topology, the script implements the `A*` algorithm.
- **Mechanics**: A* maintains a priority heap (`open_set`). It evaluates neighbouring cells by combining the known direct cost from the start (`g_score`) with a Euclidean distance heuristic to the goal.
- It guarantees the most eco-friendly optimum path that strictly avoids disrupting thick forest canopies and bypasses lakes/rivers.

### 7.3 Interactivity
The planner renders the RGB satellite map and allows the user to click interactively on two discrete points (Start and Goal). It dynamically computes the A* lineage and overlays a bright cyan trajectory vector atop the terrain.

---

## 8. Conclusion and Future Directions

The **EcoVision** project stands as a complete, multi-stage, end-to-end framework integrating spatial data engineering, robust deep convolutional networks (U-Net), and optimized classical algorithm mapping (A*). 

- **Achievements**: It successfully manages the ingestion of unyieldingly massive GEE TIFFs, intelligently scales and parses them into flawless numpy datasets, trains a segmenter with highly customized dice-bce synergy loss, and provides actionable environmental routing algorithms.
- **Scope**: The existence of `.h5` and newer `.keras` model iterations imply active continuous improvement. The modular `src` folder ensures that the dataset parser and the pathfinding engine can be decoupled or merged seamlessly into a web-serving platform.

*End of Report.*
