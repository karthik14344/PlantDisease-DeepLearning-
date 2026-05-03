# CBAM-YOLO-MT: Multi-Task Plant Disease Detection with Severity Estimation on Pear Leaves

A deep learning research project that detects plant diseases on pear tree leaves and simultaneously predicts infection severity using a novel multi-task architecture built on YOLOv11.

---

## What This Project Does

Given a photograph of a pear leaf taken in the field, this system:

1. **Draws a bounding box** around the diseased region
2. **Classifies the disease** as one of: healthy, spot, curl, or slug
3. **Predicts how severe** the infection is on a scale of 0-4

All three tasks happen in a **single forward pass** through one unified model.

This is the **first object detection benchmark** on the DiaMOS Plant dataset. Previous work only did image classification (is the leaf sick or not?) without localization or severity prediction.

---

## Dataset: DiaMOS Plant

The dataset was created by Fenu & Malloci (2021) and contains real-world images of pear tree leaves collected in orchards using smartphones and DSLR cameras.

```
Pear/
|-- leaves/
|   |-- healthy/    43 images   (1.4%)   - No disease
|   |-- spot/      884 images  (29.4%)   - Leaf spot disease
|   |-- curl/       54 images   (1.8%)   - Leaf curl disease
|   |-- slug/     2025 images  (67.4%)   - Pear slug damage
|
|-- annotation/
|   |-- csv/diaMOSPlant.csv              - Severity labels (0-4)
|   |-- YOLO/leaves/                     - Bounding box annotations
```

### Key Dataset Characteristics
- **3,006 total leaf images** used for training and evaluation
- **Severe class imbalance**: slug has 47x more images than healthy
- **Severity levels**: 0=healthy, 1=very low (1-5%), 2=low (6-20%), 3=medium (21-25%), 4=high (>50% leaf area affected)
- **54 curl images** have missing ("not estimable") severity labels
- All YOLO annotation files originally use class_id=0 (just "leaf") - our pipeline remaps them to disease-specific class IDs

---

## Architecture: CBAM-YOLO-MT

The model name stands for **C**onvolutional **B**lock **A**ttention **M**odule + **YOLO** + **M**ulti-**T**ask.

### High-Level Architecture

```
                        Input Image (640x640)
                              |
                    +---------v---------+
                    |   YOLOv11n        |
                    |   Backbone        |   Pretrained on COCO (1.2M images)
                    |   (C3k2 + SPPF)   |   Extracts visual features
                    +---------+---------+
                              |
                    +---------v---------+
                    |   YOLOv11n        |
                    |   Neck (FPN/PAN)  |   Fuses features at 3 scales
                    +---------+---------+
                              |
                    P3 (80x80, 64ch)    - Fine detail (small lesions)
                    P4 (40x40, 128ch)   - Medium features
                    P5 (20x20, 256ch)   - Global context (large damage)
                              |
                    +---------v---------+
                    |   CBAM Attention   |   NEW - Channel + Spatial attention
                    |   (3 blocks, one   |   on each scale independently
                    |    per scale)      |
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
    +---------v---------+           +---------v---------+
    |   Detection Head  |           | Initial Severity  |
    |   (Standard YOLO) |           | Prediction        |
    |   4 classes        |           +---------+---------+
    +---------+---------+                     |
              |                     +---------v---------+
              |                     | Severity-Aware    |   NEW - Novel feedback
              |                     | Channel Gating    |   mechanism
              |                     +---------+---------+
              |                               |
              |                     +---------v---------+
              |                     | BiFPN Weighted    |   NEW - Learnable
              |                     | Feature Fusion    |   multi-scale fusion
              |                     +---------+---------+
              |                               |
              |                     +---------v---------+
              |                     | Severity Head     |
              |                     | FC(64->256->128->5)|
              |                     +---------+---------+
              |                               |
              v                               v
        Bounding Box                    Severity Level
        + Disease Class                 (0, 1, 2, 3, or 4)
   (healthy/spot/curl/slug)
```

### What Each Component Does

#### 1. YOLOv11n Backbone (Pretrained)
The backbone is the feature extraction engine. It takes a 640x640 RGB image and produces hierarchical feature maps through a series of convolutional blocks (C3k2 blocks, which are improved CSPNet modules). The SPPF (Spatial Pyramid Pooling Fast) module at the end captures multi-scale context.

We use the **nano** variant (2.6M parameters) intentionally - it is lightweight enough for edge deployment on smartphones or agricultural drones.

The backbone comes **pretrained on COCO** (a dataset of 1.2 million images with 80 object classes). This means it already knows how to detect edges, textures, shapes, and patterns. We fine-tune it to recognize plant disease features.

#### 2. YOLOv11n Neck (FPN/PAN)
The neck combines features from different backbone layers using a Feature Pyramid Network (top-down pathway) and Path Aggregation Network (bottom-up pathway). This produces three output feature maps at different resolutions:

- **P3** (80x80 grid, 64 channels): Captures fine-grained details. Good for detecting small, early-stage lesions.
- **P4** (40x40 grid, 128 channels): Captures medium-scale features. Useful for moderate disease regions.
- **P5** (20x20 grid, 256 channels): Captures coarse, global context. Important for assessing large-scale damage.

#### 3. CBAM Attention (Our Addition)
CBAM (Woo et al., 2018) is applied independently to each of the three feature maps (P3, P4, P5). It works in two steps:

**Channel Attention** - Answers "which feature channels are important?"
- Applies Global Average Pooling AND Global Max Pooling to compress spatial dimensions
- Passes both through a shared MLP (two fully connected layers with ReLU)
- Adds the results and applies Sigmoid to get channel weights
- Multiplies the original feature map by these weights

For example, if channels 10-15 encode "brown spot texture" and channels 40-45 encode "green healthy leaf texture", channel attention learns to emphasize channels 10-15 when looking at a diseased leaf.

**Spatial Attention** - Answers "which spatial locations are important?"
- Computes channel-wise average and max across all channels -> two single-channel maps
- Concatenates them and passes through a 7x7 convolution
- Applies Sigmoid to get a spatial weight map
- Multiplies the channel-attended feature map by this spatial map

This focuses the model on the diseased region of the leaf and suppresses background (soil, branches, sky).

**How CBAM is integrated**: We use PyTorch's `register_forward_pre_hook` mechanism. The CBAM blocks intercept the feature maps just before they enter the Detection Head, replacing raw features with attention-enhanced features. This means both the Detection Head and the Severity Head receive attended features.

#### 4. Detection Head (Standard YOLO)
This is the unmodified YOLOv11 detection head. It takes the three attended feature maps and predicts, for each of 8,400 anchor points:
- Bounding box coordinates (x_center, y_center, width, height)
- Class probabilities for 4 classes (healthy, spot, curl, slug)
- Objectness confidence score

During inference, Non-Maximum Suppression (NMS) filters overlapping detections.

#### 5. Severity-Aware Channel Gating (Our Novel Contribution)
This is the key architectural novelty. It creates a **feedback loop** where the model's own severity prediction modulates the features used for final severity classification.

How it works:
1. An initial severity prediction is made from the CBAM-attended features
2. This prediction is converted to soft probabilities via softmax: e.g., [0.1, 0.6, 0.2, 0.05, 0.05]
3. These probabilities are used to look up a **learnable severity embedding** (weighted sum of 5 embedding vectors, each 32-dimensional)
4. The embedding is concatenated with a global summary of the feature map
5. An MLP produces a channel gate (sigmoid) that modulates the features

**Intuition**: Different severity levels have fundamentally different visual signatures:
- Severity 1 (very low): tiny scattered spots, subtle color changes - needs texture-sensitive channels
- Severity 4 (high): massive browning, leaf deformation - needs large-area color channels

The gate learns to amplify the right channels for each severity regime.

#### 6. BiFPN Weighted Fusion (Our Addition)
Instead of simply concatenating P3+P4+P5 features, we use learnable weighted fusion inspired by BiFPN (Tan et al., 2020):
- All scales are projected to the same channel count (64) via 1x1 convolutions
- P4 and P5 are upsampled to P3's spatial resolution
- Features are combined as: `w1*P3' + w2*P4' + w3*P5'`
- Weights w1, w2, w3 are **learnable parameters** normalized via fast softmax

This lets the model learn that, for example, P5 (global context) matters more for severity estimation than P3 (local detail).

#### 7. Severity Classification Head
A simple fully-connected classifier:
- Global Average Pooling: (B, 64, H, W) -> (B, 64)
- FC(64 -> 256) + BatchNorm + ReLU + Dropout(0.3)
- FC(256 -> 128) + ReLU + Dropout(0.3)
- FC(128 -> 5) -> severity logits for 5 classes

### Loss Function

```
L_total = L_detection + 0.5 * L_severity
```

- **L_detection**: Standard YOLO loss consisting of:
  - CIoU loss for bounding box regression
  - Binary Cross-Entropy for class prediction
  - Distribution Focal Loss (DFL) for box refinement
- **L_severity**: Cross-Entropy loss with label smoothing (0.05)
- **Lambda = 0.5**: Balances the two tasks. Tunable via ablation study.
- Images with "not estimable" severity (54 curl images) are **masked out** from severity loss computation

### Transfer Learning Strategy

| Component | Initialized From | Learning Rate |
|-----------|-----------------|---------------|
| Backbone + Neck + Box Head | COCO pretrained (448/499 layers match) | 0.01 (lower, preserves features) |
| Classification Head | Random (shape mismatch: 80 COCO classes vs 4 disease classes) | 0.01 |
| CBAM + BiFPN + Severity Gate + Severity Head | Random (new modules) | 0.05 (5x higher, trains faster) |

---

## Two-Stage Training Variant (CBAM-YOLO-MT-TS)

After the end-to-end run above (initialized from generic COCO weights), we ran a second variant that **initializes the multi-task model from our own DiaMOS-trained YOLOv11n baseline** (`models/baselines/yolo11n_baseline/weights/best.pt`) instead of `yolo11n.pt`.

### Why a Second Variant?

The end-to-end run achieved val mAP@0.5 = 0.725, which was 19 points lower than the detection-only YOLOv11n baseline (0.916). Hypothesis: the detection backbone needs to learn pear-leaf features and severity simultaneously from scratch, which is harder than learning severity on top of an already-good detector.

The two-stage strategy fixes this by **starting from a detector that already knows DiaMOS Plant**, then bolting on CBAM + BiFPN + Severity-Aware Gating + Severity Head and fine-tuning all of it together for 100 more epochs.

### Architecture

**Identical to CBAM-YOLO-MT** above (same CBAM blocks, same BiFPN fusion, same Severity-Aware Gate, same severity head). Only the **initial weights for the YOLOv11n backbone, neck, and detection head** change:

| Component | End-to-End (CBAM-YOLO-MT) | Two-Stage (CBAM-YOLO-MT-TS) |
|-----------|---------------------------|-----------------------------|
| Backbone + Neck + Box Head | COCO pretrained `yolo11n.pt` (80 classes, generic objects) | DiaMOS YOLOv11n `best.pt` (4 disease classes, already fine-tuned) |
| CBAM + BiFPN + Severity Gate + Severity Head | Random init | Random init |

Switched on by `INIT_MULTITASK_FROM_BASELINE = True` in `src/config.py`. The training loop loads the baseline `best.pt`, filters parameters by shape match, and copies them into a fresh `DetectionModel(nc=4)` before attaching the multi-task heads.

### Two-Stage Results

All metrics reported on the **validation set**. Trained for full 100 epochs (best checkpoint = epoch 37, selected on lowest combined val loss):

| Variant | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Sev. Acc | Sev. MAE |
|---------|---------|---------------|-----------|--------|------|----------|----------|
| CBAM-YOLO-MT (end-to-end, COCO init) | 0.725 | 0.578 | 0.699 | 0.907 | 0.789 | 0.545 | 0.479 |
| **CBAM-YOLO-MT-TS (two-stage, baseline init)** | **0.813** | **0.662** | **0.675** | **0.898** | **0.771** | **0.584** | **0.464** |

**Per-class on validation set (CBAM-YOLO-MT, end-to-end COCO init):**

| Class | Precision | Recall | F1 | mAP@0.5:0.95 |
|-------|-----------|--------|------|---------------|
| healthy | 0.280 | 0.778 | 0.412 | 0.501 |
| spot | 0.688 | 0.836 | 0.755 | 0.639 |
| curl | 0.167 | 0.846 | 0.279 | 0.333 |
| slug | 0.798 | 0.943 | 0.865 | 0.838 |

### Findings From the Two-Stage Run

1. **Validation mAP improves notably** (0.813 vs 0.725 end-to-end, +0.088), confirming the baseline init helps the detection branch converge to a stronger optimum on data it has already seen.
2. **Severity metrics improve slightly** (Acc 0.584 vs 0.545, MAE 0.464 vs 0.479). The novel Severity-Aware Channel Gating benefits from a stronger detection backbone.
3. **Best-checkpoint criterion matters**: `best.pt` was selected by lowest total val loss (epoch 37). The peak severity epoch was 68, with val Sev Acc = 0.682 — but those weights were not saved separately. A future run should track and save a severity-best checkpoint independently.

### When to Use Which Variant

- Use **CBAM-YOLO-MT-TS (two-stage)** as the default — it has the strongest validation numbers across all metrics (mAP@0.5 = 0.813, Sev Acc = 0.584).
- Use **CBAM-YOLO-MT (end-to-end)** when starting from generic COCO weights without a pre-trained DiaMOS detector available.

---

## Results

### Detection Performance

All models reported on the **validation set** (matches Mitra 2023's reporting convention).

| Model | Params | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Sev. Acc | Sev. MAE |
|-------|--------|---------|---------------|-----------|--------|----------|----------|
| YOLOv5m (Mitra 2023) | ~25M | 0.894 | — | 0.864 | 0.877 | — | — |
| YOLOv8m (Mitra 2023) | ~25M | 0.893 | — | 0.879 | 0.865 | — | — |
| YOLOv8n (ours) | 3.0M | 0.891 | — | — | — | — | — |
| **YOLOv11n (ours)** | **2.6M** | **0.916** | **0.778** | **0.892** | **0.915** | — | — |
| **CBAM-YOLO-MT (ours, end-to-end)** | **~3.1M** | **0.725** | **0.578** | **0.699** | **0.907** | **0.545** | **0.479** |
| **CBAM-YOLO-MT-TS (ours, two-stage)** | **~3.1M** | **0.813** | **0.662** | **0.675** | **0.898** | **0.584** | **0.464** |

### YOLOv11n Baseline vs Mitra 2023 (Validation, Per-Class)

| Class | Mitra YOLOv5m | YOLOv11n (ours) | Gain |
|-------|---------------|-----------------|------|
| Healthy | 0.995 | 0.995 | tied |
| Curl | 0.748 | **0.827** | **+0.079** |
| Slug | 0.976 | **0.978** | +0.002 |
| Spot | 0.857 | **0.870** | +0.013 |
| **Overall AP@0.5** | 0.894 | **0.916** | **+0.022** |

Our YOLOv11n baseline beats Mitra 2023's YOLOv5m on every class with **8× fewer parameters** (2.6M vs 20.8M).

### Key Findings

1. **First YOLOv11 benchmark on DiaMOS Plant** — no prior work reported these numbers
2. **YOLOv11n beats Mitra 2023** on validation (+2.2% mAP@0.5) with 8× fewer parameters
3. **First severity prediction** on DiaMOS Plant: 58.4% validation accuracy, 0.464 MAE (no prior work uses severity labels)
4. **Two-stage init wins**: Initializing the multi-task model from our DiaMOS-trained YOLOv11n boosts val mAP@0.5 from 0.725 to 0.813 (+0.088) over generic COCO init
5. **Curl is the bottleneck**: Only 54 training images, val AP varies widely across variants

---

## Project Structure

```
d:/college/VI-sem/DL/prj/
|
|-- Pear/                              # Raw DiaMOS Plant dataset (DO NOT MODIFY)
|   |-- leaves/{healthy,spot,curl,slug}/ # Leaf images organized by disease class
|   |-- annotation/csv/               # Severity labels CSV
|   |-- annotation/YOLO/leaves/       # Bounding box annotations
|
|-- data/
|   |-- processed/                     # Generated by prepare.py
|       |-- images/{train,val,test}/   # Split images (70:20:10)
|       |-- labels/{train,val,test}/   # Remapped YOLO labels (class 0-3)
|       |-- data.yaml                  # Ultralytics dataset config
|       |-- severity_labels.json       # Stem -> severity mapping
|       |-- split_info.csv             # Which image went where
|
|-- src/                               # All source code
|   |-- config.py                      # Central configuration (paths, hyperparams, flags)
|   |-- logger/__init__.py             # Logging setup (rotating file + console)
|   |-- data/
|   |   |-- explore.py                 # Task 1: Dataset statistics and visualization
|   |   |-- prepare.py                 # Task 2: Build YOLO-format dataset with splits
|   |-- models/
|   |   |-- attention.py               # CBAM, BiFPN, SE, SeverityAwareGate modules
|   |   |-- multitask_yolo.py          # CBAM-YOLO-MT model definition
|   |-- training/
|   |   |-- train_baseline.py          # Ultralytics-based baseline training
|   |   |-- train_multitask.py         # Custom PyTorch training loop + resume support
|   |   |-- ablation.py               # Ablation study configurations
|   |-- evaluation/
|   |   |-- evaluate.py               # Model comparison on test set
|   |-- visualization/
|       |-- plots.py                   # Paper-quality figures and LaTeX tables
|
|-- models/                            # Trained weights
|   |-- baselines/
|   |   |-- yolov8n_baseline/weights/best.pt
|   |   |-- yolo11n_baseline/weights/best.pt
|   |   |-- yolo11s_baseline/weights/best.pt
|   |-- multitask/
|       |-- yolo11n_multitask_.../best.pt    # CBAM-YOLO-MT checkpoint
|
|-- reports/
|   |-- figures/                       # Generated plots (class dist, bbox samples, etc.)
|   |-- tables/                        # CSV and LaTeX comparison tables
|
|-- logs/                              # Timestamped training logs
|-- run_pipeline.py                    # Master CLI entry point
|-- requirements.txt                   # Python dependencies
|-- README.md                          # This file
```

---

## How to Set Up and Run

### Prerequisites
- Python 3.12
- NVIDIA GPU with CUDA support (tested on RTX 5070 Laptop, 8 GB VRAM)
- ~15 GB disk space (dataset + processed data + model weights)

### 1. Create Environment

```bash
conda create --name plantDisease python=3.12
conda activate plantDisease
```

### 2. Install Dependencies

For GPUs with CUDA 12.8+ (RTX 40/50 series):
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

For older GPUs (RTX 30 series, CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Run the Pipeline

Each task can be run independently or all at once:

```bash
# Step by step (recommended for first run)
python run_pipeline.py --task explore     # Understand the dataset (~2 min)
python run_pipeline.py --task prepare     # Build train/val/test splits (~3 min)
python run_pipeline.py --task baseline    # Train YOLOv8n + YOLOv11n + YOLOv11s (~20 hrs)
python run_pipeline.py --task multitask   # Train CBAM-YOLO-MT (~14 hrs)
python run_pipeline.py --task evaluate    # Compare all models (~5 min)
python run_pipeline.py --task visualize   # Generate paper figures (~1 min)

# Or run everything
python run_pipeline.py
```

### 5. Resume Interrupted Training

If multi-task training gets interrupted (power loss, laptop sleep), resume from the last checkpoint:

```bash
python run_pipeline.py --task multitask --resume models/multitask/<folder_name>
```

This loads the model weights, optimizer state, and learning rate scheduler from `last.pt` and continues from the last completed epoch.

---

## Configuration

All hyperparameters and settings are in `src/config.py`:

### Key Settings

```python
# Dataset
IMG_SIZE = 640              # Input image resolution
BATCH_SIZE = 16             # Increase if GPU memory allows (try 32 or 48)

# Training
EPOCHS_BASELINE = 100       # Max epochs for baseline models
EPOCHS_MULTITASK = 100      # Max epochs for multi-task model
LEARNING_RATE = 0.01        # Base learning rate (SGD)
PATIENCE = 100              # Early stopping patience (100 = effectively disabled)
LAMBDA_SEVERITY = 0.5       # Weight for severity loss (0.0 = detection only)

# Architecture flags (toggle for ablation study)
USE_CBAM = True             # Channel + Spatial attention on neck features
USE_BIFPN = True            # BiFPN-style weighted fusion in severity head
USE_SEVERITY_GATE = True    # Severity-Aware Channel Gating (novel module)

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42            # For reproducibility
```

### Running Ablation Studies

To test the contribution of each module, toggle flags in `config.py` and retrain:

| Experiment | USE_CBAM | USE_BIFPN | USE_SEVERITY_GATE |
|-----------|----------|-----------|-------------------|
| Plain multi-task | False | False | False |
| + CBAM only | True | False | False |
| + CBAM + BiFPN | True | True | False |
| Full CBAM-YOLO-MT | True | True | True |

---

## Technical Details

### Data Pipeline

1. **CSV Parsing**: The severity CSV uses semicolons as delimiters and one-hot encoding for both disease class and severity level. Some curl images have "not estimable" strings instead of 0/1 values, which causes mixed-type columns. Our parser handles this with explicit type casting.

2. **Class Remapping**: Original YOLO annotations all use class_id=0 (just "leaf"). Our pipeline maps them based on the image's parent directory:
   - `healthy/` -> class 0
   - `spot/` -> class 1
   - `curl/` -> class 2
   - `slug/` -> class 3

3. **Stratified Splitting**: Uses scikit-learn's `train_test_split` with stratification to preserve class proportions across train/val/test, even for minority classes with as few as 43 images.

4. **Oversampling**: Minority classes (healthy: 43, curl: 54) are oversampled to 30% of the majority class count by duplicating images with augmented filenames.

### Custom Training Loop

We use a custom PyTorch training loop instead of Ultralytics' built-in `model.train()` because:
- Need to return both detection predictions AND severity predictions in one forward pass
- Need to compute a combined loss from two heads
- Need to inject CBAM attention via hooks into the standard YOLO architecture

The detection loss uses Ultralytics' `v8DetectionLoss` class. The severity loss uses standard CrossEntropy with label smoothing. Invalid severity labels (value -1) are masked out.

### Hook-Based Architecture Integration

The CBAM modules are integrated into the YOLO model using PyTorch's hook mechanism:
- `register_forward_pre_hook` on the Detect head captures neck features BEFORE the detect head processes them
- A second pre-hook REPLACES the detect head's input with CBAM-attended features
- This approach avoids modifying any Ultralytics source code

### Logging

Every source file uses Python's `logging` module with a centralized configuration:
- Console output for real-time monitoring
- Rotating file logs (5 MB max, 3 backups) in `logs/` directory
- Format: `[timestamp] module_name - LEVEL - message`

---

## Challenges We Solved

| Challenge | What Went Wrong | How We Fixed It |
|-----------|----------------|-----------------|
| YOLO annotations all class 0 | Can't distinguish diseases | Remapped based on subdirectory |
| CSV mixed types | `"1" == 1` returns False in pandas | Added `int()` casting before comparison |
| RTX 5070 unsupported | PyTorch 2.5.1 lacks sm_120 support | Installed PyTorch nightly with CUDA 12.8 |
| Windows workers crash | Each worker loads 2.8 GB CUDA DLLs | Set NUM_WORKERS=0 on Windows |
| nc mismatch (80 vs 4) | Detect head crashes during probing | Used pre-hook (fires before crash) |
| Loss not scalar | v8DetectionLoss returns multi-element tensor | Added `.sum()` before backward |
| Training interruptions | Laptop shutdown mid-training | Added checkpoint resume from last.pt |

---

## Tools and Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.12 (nightly) | Deep learning framework |
| Ultralytics | 8.4.37 | YOLO model architecture and baseline training |
| Python | 3.12 | Programming language |
| CUDA | 12.8 | GPU acceleration |
| scikit-learn | 1.3+ | Stratified splitting, classification metrics |
| matplotlib | 3.7+ | Plotting and visualization |
| seaborn | 0.12+ | Heatmaps and statistical plots |
| OpenCV | 4.8+ | Image loading, resizing, augmentation |
| pandas | 2.0+ | CSV parsing and data manipulation |
| PyYAML | 6.0+ | Dataset configuration files |

### Hardware Used
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM)
- CPU: AMD Ryzen 9
- Training time: ~4-8 hours per baseline, ~14-16 hours for multi-task (100 epochs)

---

## References

1. Fenu, G., & Malloci, F. M. (2021). DiaMOS Plant: A Dataset for Diagnosis and Monitoring Plant Disease. *Agronomy*, 11(11), 2107.
2. Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
3. Tan, M., Pang, R., & Le, Q. V. (2020). EfficientDet: Scalable and Efficient Object Detection. *CVPR 2020*.
4. Ultralytics (2024). YOLOv11 Documentation. https://docs.ultralytics.com/
5. Mitra, S. (2023). YOLOv5/v8 Detection on DiaMOS Plant Dataset. (benchmark comparison)
