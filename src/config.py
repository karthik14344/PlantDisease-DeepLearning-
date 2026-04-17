"""
Central configuration for the DiaMOS Plant Disease Detection project.
All paths, hyperparameters, and constants in one place.
"""
import os
import logging
import torch
from pathlib import Path

from src.logger import configure_logger
configure_logger()

logger = logging.getLogger(__name__)

# ── Project Root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Pear"

# ── Dataset Paths ──
LEAVES_DIR = DATA_ROOT / "leaves"
FRUITS_DIR = DATA_ROOT / "fruits"
YOLO_ANNOT_DIR = DATA_ROOT / "annotation" / "YOLO" / "leaves"
CSV_PATH = DATA_ROOT / "annotation" / "csv" / "diaMOSPlant.csv"

# ── Output Paths ──
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# ── Class Definitions ──
CLASS_NAMES = ["healthy", "spot", "curl", "slug"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

# Map CSV column names → our class names
CSV_COL_TO_CLASS = {
    "healthy": "healthy",
    "leaf_spot": "spot",
    "curl": "curl",
    "pear_slug": "slug",
}

# Map subdirectory names → class indices
SUBDIR_TO_CLASS_IDX = {
    "healthy": 0,
    "spot": 1,
    "curl": 2,
    "slug": 3,
}

# ── Severity Definitions ──
SEVERITY_NAMES = ["healthy_0", "very_low_1", "low_2", "medium_3", "high_4"]
NUM_SEVERITY = 5
SEVERITY_NOT_ESTIMABLE = -1  # Placeholder for curl images with unknown severity

# ── Data Split ──
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

# ── Training Hyperparameters ──
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS_BASELINE = 100
EPOCHS_MULTITASK = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
PATIENCE = 100  # Effectively disabled — run full EPOCHS_MULTITASK

# Multi-task
LAMBDA_SEVERITY = 0.5  # Weight for severity loss
SEVERITY_LOSS_TYPE = "ce"  # "ce" for CrossEntropy, "mse" for regression

# Architectural upgrades (CBAM-YOLO-MT)
USE_CBAM = True              # Channel + Spatial attention on neck features
USE_BIFPN = True             # BiFPN-style weighted fusion in severity head
USE_SEVERITY_GATE = True     # Severity-Aware Channel Gating (novel)

# Augmentation
MOSAIC_PROB = 1.0
COPY_PASTE_PROB = 0.3
MIXUP_PROB = 0.1
FLIP_LR_PROB = 0.5
FLIP_UD_PROB = 0.1
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 2.0

# ── Device ──
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    NUM_GPUS = torch.cuda.device_count()
    logger.info(f"GPU detected: {GPU_NAME} ({GPU_MEM:.1f} GB), count={NUM_GPUS}")
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "N/A"
    GPU_MEM = 0
    NUM_GPUS = 0
    logger.warning("No GPU detected, using CPU")

NUM_WORKERS = 0 if os.name == "nt" else min(8, os.cpu_count() or 4)

# ── Model configs ──
BASELINE_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
}

ABLATION_LAMBDAS = [0.1, 0.3, 0.5, 0.7, 1.0]

logger.info(f"Config loaded | device={DEVICE} | classes={NUM_CLASSES} | img_size={IMG_SIZE}")
