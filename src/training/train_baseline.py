"""
Task 4a: Baseline Training
- Train YOLOv8n and YOLOv11n on the prepared dataset
- Standard Ultralytics training API
- Establishes detection-only mAP benchmarks
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PROCESSED_DIR, MODELS_DIR, NUM_CLASSES, CLASS_NAMES,
    IMG_SIZE, BATCH_SIZE, EPOCHS_BASELINE, LEARNING_RATE,
    PATIENCE, DEVICE, NUM_WORKERS, BASELINE_MODELS, NUM_GPUS,
    MOSAIC_PROB, COPY_PASTE_PROB, MIXUP_PROB, FLIP_LR_PROB,
    HSV_H, HSV_S, HSV_V, DEGREES, TRANSLATE, SCALE, SHEAR,
)

logger = logging.getLogger(__name__)


def train_baseline_model(model_name, model_weights, data_yaml, output_name=None):
    """Train a single baseline YOLO model using Ultralytics API."""
    from ultralytics import YOLO

    logger.info("=" * 60)
    logger.info(f"TRAINING BASELINE: {model_name}")
    logger.info(f"  weights={model_weights}, data={data_yaml}")

    model = YOLO(model_weights)

    run_name = output_name or f"{model_name}_baseline_{datetime.now():%Y%m%d_%H%M}"
    project_dir = str(MODELS_DIR / "baselines")

    train_args = dict(
        data=str(data_yaml),
        epochs=EPOCHS_BASELINE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="0" if str(DEVICE) == "cuda" else "cpu",
        workers=NUM_WORKERS,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=LEARNING_RATE,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=PATIENCE,
        mosaic=MOSAIC_PROB,
        copy_paste=COPY_PASTE_PROB,
        mixup=MIXUP_PROB,
        fliplr=FLIP_LR_PROB,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    logger.info(f"Starting training: epochs={EPOCHS_BASELINE}, batch={BATCH_SIZE}, imgsz={IMG_SIZE}")
    results = model.train(**train_args)

    metrics_path = Path(project_dir) / run_name / "metrics_summary.json"
    try:
        # Ultralytics returns DetMetrics; use results_dict for metrics
        rd = results.results_dict
        metrics = {
            "model": model_name,
            "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)", 0)),
            "recall": float(rd.get("metrics/recall(B)", 0)),
        }
        # Per-class AP if available
        try:
            ap50_list = results.box.ap50.tolist()
            for i, cls_name in enumerate(CLASS_NAMES):
                if i < len(ap50_list):
                    metrics[f"AP50_{cls_name}"] = float(ap50_list[i])
        except Exception:
            pass
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")
        logger.info(f"  mAP50={metrics['mAP50']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    except Exception as e:
        logger.exception(f"Could not save metrics summary: {e}")

    return results


def validate_baseline_model(model_weights_path, data_yaml):
    """Run validation on a trained model."""
    from ultralytics import YOLO

    logger.info(f"Validating model: {model_weights_path}")
    model = YOLO(model_weights_path)
    results = model.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="0" if str(DEVICE) == "cuda" else "cpu",
    )
    return results


def run_baseline_training():
    """Train all baseline models."""
    logger.info("#" * 60)
    logger.info("TASK 4a: BASELINE TRAINING")
    logger.info("#" * 60)

    data_yaml = PROCESSED_DIR / "data.yaml"
    if not data_yaml.exists():
        logger.error(f"data.yaml not found at {data_yaml}. Run prepare.py first.")
        return

    all_results = {}

    for model_name, weights in BASELINE_MODELS.items():
        try:
            results = train_baseline_model(
                model_name=model_name,
                model_weights=weights,
                data_yaml=data_yaml,
                output_name=f"{model_name}_baseline",
            )
            all_results[model_name] = results
            logger.info(f"{model_name} training complete!")
        except Exception as e:
            logger.exception(f"ERROR training {model_name}: {e}")

    logger.info("=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    return all_results


if __name__ == "__main__":
    run_baseline_training()
