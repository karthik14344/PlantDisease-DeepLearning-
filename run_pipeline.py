"""
Master Pipeline Script for DiaMOS Plant Disease Detection Project
=================================================================

Multi-Task YOLOv11 for Pear Tree Disease Detection with Severity Prediction.

Usage:
    python run_pipeline.py                  # Run everything
    python run_pipeline.py --task explore   # Run only exploration
    python run_pipeline.py --task prepare   # Run only data preparation
    python run_pipeline.py --task baseline  # Train baselines only
    python run_pipeline.py --task multitask # Train multi-task model
    python run_pipeline.py --task evaluate  # Evaluate all models
    python run_pipeline.py --task ablation  # Run ablation studies
    python run_pipeline.py --task visualize # Generate paper outputs
"""
import os
import sys
import logging
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import configure_logger
configure_logger()

logger = logging.getLogger(__name__)


def log_device_info():
    from src.config import DEVICE, GPU_NAME, GPU_MEM, NUM_GPUS
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {GPU_NAME} ({GPU_MEM:.1f} GB), count={NUM_GPUS}")


def task_explore():
    from src.data.explore import run_exploration
    return run_exploration()

def task_prepare():
    from src.data.prepare import run_preparation
    return run_preparation()

def task_baseline():
    from src.training.train_baseline import run_baseline_training
    return run_baseline_training()

def task_multitask(resume_dir=None):
    from src.training.train_multitask import run_multitask_training
    return run_multitask_training(resume_dir=resume_dir)

def task_evaluate():
    from src.evaluation.evaluate import run_evaluation
    return run_evaluation()

def task_ablation():
    from src.training.ablation import run_all_ablations
    return run_all_ablations()

def task_visualize():
    from src.visualization.plots import run_visualization
    return run_visualization()


def run_all():
    logger.info("=" * 70)
    logger.info("DiaMOS Plant Disease Detection - Full Pipeline")
    logger.info("=" * 70)
    log_device_info()

    for phase, name, fn in [
        (1, "DATASET EXPLORATION", task_explore),
        (2, "DATA PREPARATION", task_prepare),
        (3, "BASELINE TRAINING", task_baseline),
        (4, "MULTI-TASK TRAINING", task_multitask),
        (5, "EVALUATION", task_evaluate),
        (6, "ABLATION STUDIES", task_ablation),
        (7, "PAPER-READY OUTPUTS", task_visualize),
    ]:
        logger.info(f">>> PHASE {phase}: {name}")
        try:
            fn()
        except Exception as e:
            logger.exception(f"PHASE {phase} FAILED: {e}")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE! Check reports/figures/ and reports/tables/")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="DiaMOS Plant Disease Detection Pipeline")
    parser.add_argument(
        "--task",
        choices=["explore", "prepare", "baseline", "multitask",
                 "evaluate", "ablation", "visualize", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to multitask run folder to resume training from last.pt",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("DiaMOS Plant Disease Detection - Multi-Task YOLOv11 Pipeline")
    logger.info("=" * 70)
    log_device_info()

    if args.task == "multitask":
        task_multitask(resume_dir=args.resume)
    else:
        task_map = {
            "explore": task_explore, "prepare": task_prepare,
            "baseline": task_baseline,
            "evaluate": task_evaluate, "ablation": task_ablation,
            "visualize": task_visualize, "all": run_all,
        }
        task_map[args.task]()


if __name__ == "__main__":
    main()
