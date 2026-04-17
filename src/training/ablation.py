"""
Task 6: Ablation Studies
- With vs without severity head
- With vs without minority class augmentation
- Different lambda values for loss weighting
"""
import sys
import json
import logging
import shutil
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PROCESSED_DIR, MODELS_DIR, TABLES_DIR,
    ABLATION_LAMBDAS, EPOCHS_MULTITASK, IMG_SIZE, BATCH_SIZE,
    DEVICE, LAMBDA_SEVERITY, CLASS_NAMES,
)

logger = logging.getLogger(__name__)


def ablation_lambda_sweep():
    from src.training.train_multitask import run_multitask_training

    logger.info("#" * 60)
    logger.info("ABLATION: Lambda Sweep")

    results = []
    for lam in ABLATION_LAMBDAS:
        logger.info(f"--- Lambda = {lam} ---")
        run_name = f"ablation_lambda_{lam}"
        try:
            output_dir, history = run_multitask_training(
                model_name="yolo11n.pt", lambda_sev=lam, run_name=run_name,
            )
            best_val = min(history["val"], key=lambda x: x["total_loss"])
            results.append({
                "lambda": lam,
                "best_val_loss": best_val["total_loss"],
                "best_det_loss": best_val["det_loss"],
                "best_sev_loss": best_val["sev_loss"],
                "best_sev_acc": best_val.get("sev_acc", 0),
                "best_sev_mae": best_val.get("sev_mae", 0),
                "epochs_trained": len(history["train"]),
            })
            logger.info(f"Lambda={lam}: val_loss={best_val['total_loss']:.4f}, sev_acc={best_val.get('sev_acc', 0):.4f}")
        except Exception as e:
            logger.exception(f"ERROR with lambda={lam}: {e}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = TABLES_DIR / "ablation_lambda.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Lambda ablation results saved: {csv_path}")
    return results


def ablation_with_without_severity():
    from src.training.train_multitask import run_multitask_training

    logger.info("#" * 60)
    logger.info("ABLATION: With vs Without Severity Head")

    results = []
    for lam, label in [(0.0, "detection_only"), (LAMBDA_SEVERITY, "multitask")]:
        logger.info(f"Running: {label} (lambda={lam})")
        try:
            output_dir, history = run_multitask_training(
                model_name="yolo11n.pt", lambda_sev=lam, run_name=f"ablation_{label}",
            )
            best_val = min(history["val"], key=lambda x: x["total_loss"])
            results.append({
                "config": label, "lambda": lam,
                "best_val_loss": best_val["total_loss"],
                "best_det_loss": best_val["det_loss"],
                "best_sev_acc": best_val.get("sev_acc", 0),
            })
        except Exception as e:
            logger.exception(f"ERROR in {label}: {e}")

    df = pd.DataFrame(results)
    csv_path = TABLES_DIR / "ablation_severity_head.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Severity head ablation saved: {csv_path}")
    return results


def ablation_augmentation():
    from src.data.prepare import (
        build_image_annotation_list, stratified_split,
        oversample_minority_classes, write_split_to_disk, generate_data_yaml,
    )
    from src.training.train_baseline import train_baseline_model

    logger.info("#" * 60)
    logger.info("ABLATION: With vs Without Augmentation")

    records = build_image_annotation_list()
    train_recs, val_recs, test_recs = stratified_split(records)

    results = []
    for use_aug, label in [(False, "no_oversample"), (True, "with_oversample")]:
        logger.info(f"Config: {label}")

        aug_dir = PROCESSED_DIR / f"ablation_{label}"
        if aug_dir.exists():
            shutil.rmtree(aug_dir)
        aug_dir.mkdir(parents=True)

        train_data = oversample_minority_classes(train_recs, target_ratio=0.3) if use_aug else train_recs
        write_split_to_disk(train_data, "train", aug_dir)
        write_split_to_disk(val_recs, "val", aug_dir)
        write_split_to_disk(test_recs, "test", aug_dir)
        yaml_path = generate_data_yaml(aug_dir)

        try:
            result = train_baseline_model(
                model_name=f"yolo11n_{label}", model_weights="yolo11n.pt",
                data_yaml=yaml_path, output_name=f"ablation_{label}",
            )
            results.append({
                "config": label,
                "mAP50": float(result.results_dict.get("metrics/mAP50(B)", 0)),
                "precision": float(result.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(result.results_dict.get("metrics/recall(B)", 0)),
            })
        except Exception as e:
            logger.exception(f"ERROR in {label}: {e}")

    df = pd.DataFrame(results)
    csv_path = TABLES_DIR / "ablation_augmentation.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Augmentation ablation saved: {csv_path}")
    return results


def run_all_ablations():
    logger.info("#" * 60)
    logger.info("TASK 6: ABLATION STUDIES")
    logger.info("#" * 60)

    all_results = {}

    logger.info("[1/3] Lambda sweep...")
    all_results["lambda_sweep"] = ablation_lambda_sweep()

    logger.info("[2/3] With vs without severity head...")
    all_results["severity_head"] = ablation_with_without_severity()

    logger.info("[3/3] With vs without augmentation...")
    all_results["augmentation"] = ablation_augmentation()

    combined_path = TABLES_DIR / "all_ablation_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("ALL ABLATION STUDIES COMPLETE")
    return all_results


if __name__ == "__main__":
    run_all_ablations()
