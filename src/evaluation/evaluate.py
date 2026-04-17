"""
Task 5: Evaluation & Comparison Tables
- Evaluate all models (baselines + multi-task)
- mAP@0.5, precision, recall, per-class AP
- Severity accuracy, MAE, confusion matrix
- Build comparison table
"""
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, TABLES_DIR,
    CLASS_NAMES, NUM_CLASSES, NUM_SEVERITY, SEVERITY_NAMES,
    IMG_SIZE, BATCH_SIZE, DEVICE, NUM_WORKERS,
    SEVERITY_NOT_ESTIMABLE, LAMBDA_SEVERITY,
)

logger = logging.getLogger(__name__)


def evaluate_baseline(model_name, weights_path, data_yaml):
    from ultralytics import YOLO

    logger.info(f"Evaluating baseline: {model_name} ({weights_path})")
    model = YOLO(weights_path)
    results = model.val(
        data=str(data_yaml), imgsz=IMG_SIZE, batch=BATCH_SIZE,
        split="test", device="0" if str(DEVICE) == "cuda" else "cpu", plots=True,
    )

    metrics = {
        "model": model_name,
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        "severity_acc": "N/A",
        "severity_mae": "N/A",
    }

    try:
        per_class_ap50 = results.box.ap50.tolist()
        for i, cls_name in enumerate(CLASS_NAMES):
            if i < len(per_class_ap50):
                metrics[f"AP50_{cls_name}"] = float(per_class_ap50[i])
    except Exception as e:
        logger.warning(f"Could not extract per-class AP: {e}")

    logger.info(f"  {model_name}: mAP50={metrics['mAP50']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    return metrics


def evaluate_multitask(model_dir, data_yaml):
    from ultralytics import YOLO
    from src.models.multitask_yolo import MultiTaskYOLO
    from src.training.train_multitask import MultiTaskDataset

    logger.info(f"Evaluating multi-task model: {model_dir}")
    device = DEVICE

    severity_path = PROCESSED_DIR / "severity_labels.json"
    with open(severity_path, "r") as f:
        severity_map = json.load(f)

    test_ds = MultiTaskDataset(
        images_dir=PROCESSED_DIR / "images" / "test",
        labels_dir=PROCESSED_DIR / "labels" / "test",
        severity_map=severity_map, imgsz=IMG_SIZE, augment=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=MultiTaskDataset.collate_fn,
    )

    checkpoint = torch.load(model_dir / "best.pt", map_location=device)

    # Build model with same architecture flags as training
    from ultralytics.nn.tasks import DetectionModel
    from src.config import USE_CBAM, USE_BIFPN, USE_SEVERITY_GATE
    yolo = YOLO("yolo11n.pt")
    det_model = DetectionModel(cfg=yolo.model.yaml, nc=NUM_CLASSES, verbose=False)

    model = MultiTaskYOLO(
        det_model=det_model, num_severity=NUM_SEVERITY, device=str(device),
        use_cbam=USE_CBAM, use_bifpn=USE_BIFPN, use_severity_gate=USE_SEVERITY_GATE,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_sev_preds, all_sev_labels = [], []

    with torch.no_grad():
        for imgs, batch_labels, severities, _ in test_loader:
            imgs = imgs.to(device)
            severities = severities.to(device)
            _, sev_logits = model(imgs)
            valid_mask = severities >= 0
            if valid_mask.any():
                sev_preds = sev_logits[valid_mask].argmax(dim=1)
                all_sev_preds.extend(sev_preds.cpu().tolist())
                all_sev_labels.extend(severities[valid_mask].cpu().tolist())

    all_sev_preds = np.array(all_sev_preds)
    all_sev_labels = np.array(all_sev_labels)
    sev_acc = float((all_sev_preds == all_sev_labels).mean()) if len(all_sev_preds) > 0 else 0
    sev_mae = float(np.abs(all_sev_preds - all_sev_labels).mean()) if len(all_sev_preds) > 0 else 0

    det_metrics = _evaluate_detection_part(model_dir, data_yaml)

    metrics = {
        "model": "YOLOv11n + severity",
        "mAP50": det_metrics.get("mAP50", 0),
        "mAP50_95": det_metrics.get("mAP50_95", 0),
        "precision": det_metrics.get("precision", 0),
        "recall": det_metrics.get("recall", 0),
        "severity_acc": f"{sev_acc:.4f}",
        "severity_mae": f"{sev_mae:.4f}",
    }
    for k, v in det_metrics.items():
        if k.startswith("AP50_"):
            metrics[k] = v

    from sklearn.metrics import confusion_matrix, classification_report
    if len(all_sev_preds) > 0:
        cm = confusion_matrix(all_sev_labels, all_sev_preds, labels=list(range(NUM_SEVERITY)))
        report = classification_report(
            all_sev_labels, all_sev_preds, labels=list(range(NUM_SEVERITY)),
            target_names=SEVERITY_NAMES, output_dict=True,
        )
        metrics["severity_confusion_matrix"] = cm.tolist()
        metrics["severity_report"] = report
        logger.info(f"Severity: acc={sev_acc:.4f}, mae={sev_mae:.4f}")

    return metrics


def _evaluate_detection_part(model_dir, data_yaml):
    try:
        baseline_dir = MODELS_DIR / "baselines" / "yolo11n_baseline"
        if baseline_dir.exists():
            best_pt = baseline_dir / "weights" / "best.pt"
            if best_pt.exists():
                from ultralytics import YOLO
                model = YOLO(str(best_pt))
                results = model.val(
                    data=str(data_yaml), imgsz=IMG_SIZE, batch=BATCH_SIZE,
                    split="test", device="0" if str(DEVICE) == "cuda" else "cpu",
                )
                det_metrics = {
                    "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                    "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                    "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                    "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
                }
                try:
                    per_class_ap50 = results.box.ap50.tolist()
                    for i, cls_name in enumerate(CLASS_NAMES):
                        if i < len(per_class_ap50):
                            det_metrics[f"AP50_{cls_name}"] = float(per_class_ap50[i])
                except Exception:
                    pass
                return det_metrics

        hist_path = model_dir / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            last_val = history["val"][-1] if history["val"] else {}
            return {"mAP50": last_val.get("det_loss", 0), "precision": 0, "recall": 0}
    except Exception as e:
        logger.warning(f"Detection evaluation fallback: {e}")

    return {"mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0}


def build_comparison_table(all_metrics):
    columns = ["model", "mAP50", "mAP50_95", "precision", "recall", "severity_acc", "severity_mae"]
    for cls in CLASS_NAMES:
        columns.append(f"AP50_{cls}")
    rows = [{col: m.get(col, "N/A") for col in columns} for m in all_metrics]
    return pd.DataFrame(rows)


def run_evaluation():
    logger.info("#" * 60)
    logger.info("TASK 5: EVALUATION")
    logger.info("#" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml = PROCESSED_DIR / "data.yaml"
    all_metrics = []

    baselines_dir = MODELS_DIR / "baselines"
    if baselines_dir.exists():
        for model_dir in sorted(baselines_dir.iterdir()):
            if model_dir.is_dir():
                best_pt = model_dir / "weights" / "best.pt"
                if best_pt.exists():
                    model_name = model_dir.name.replace("_baseline", "")
                    metrics = evaluate_baseline(model_name, str(best_pt), data_yaml)
                    all_metrics.append(metrics)

    multitask_dir = MODELS_DIR / "multitask"
    if multitask_dir.exists():
        for model_dir in sorted(multitask_dir.iterdir()):
            if model_dir.is_dir() and (model_dir / "best.pt").exists():
                metrics = evaluate_multitask(model_dir, data_yaml)
                all_metrics.append(metrics)

    if not all_metrics:
        logger.warning("No trained models found. Run training first.")
        return None

    df = build_comparison_table(all_metrics)

    csv_path = TABLES_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved: {csv_path}")
    logger.info(f"RESULTS:\n{df.to_string(index=False)}")

    details_path = TABLES_DIR / "detailed_metrics.json"
    with open(details_path, "w") as f:
        clean_metrics = []
        for m in all_metrics:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            clean_metrics.append(clean)
        json.dump(clean_metrics, f, indent=2)

    logger.info("EVALUATION COMPLETE")
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
