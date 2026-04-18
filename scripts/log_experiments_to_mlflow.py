"""
Retroactive MLflow Logging Script
=================================

Reads all existing training outputs from `models/baselines/` and `models/multitask/`
and logs them as runs on the DagsHub MLflow server.

Logs:
  - Hyperparameters (from config.py + args.yaml)
  - Final metrics (from metrics_summary.json / comparison_table.csv)
  - Per-epoch metrics (from results.csv / training_history.json)
  - Artifacts (best.pt, confusion matrices, training curves)
  - Tags (architecture type, dataset)

Usage:
    python scripts/log_experiments_to_mlflow.py
"""
import sys
import os
import json
import csv
from pathlib import Path

import re
import math
import dagshub
import mlflow
import yaml


def clean_metric_name(name: str) -> str:
    """Sanitize a metric name for MLflow (no parens, replace / with _)."""
    name = name.strip()
    # MLflow allows: alphanumerics, _, -, ., space, /, :
    # Remove parens and other special chars
    name = re.sub(r"[()]", "", name)
    name = name.replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_\-\.: ]", "_", name)
    return name


def safe_log_metric(key: str, value, step: int = None):
    """Log a metric only if value is a finite number."""
    try:
        v = float(value)
        if not math.isfinite(v):
            return False
        key = clean_metric_name(key)
        if step is not None:
            mlflow.log_metric(key, v, step=step)
        else:
            mlflow.log_metric(key, v)
        return True
    except (ValueError, TypeError):
        return False

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR, TABLES_DIR, CLASS_NAMES,
    IMG_SIZE, BATCH_SIZE, EPOCHS_BASELINE, EPOCHS_MULTITASK,
    LEARNING_RATE, WEIGHT_DECAY, LAMBDA_SEVERITY,
    USE_CBAM, USE_BIFPN, USE_SEVERITY_GATE,
    MOSAIC_PROB, COPY_PASTE_PROB, MIXUP_PROB,
)

DAGSHUB_OWNER = "karthik14344"
DAGSHUB_REPO = "PlantDisease-DeepLearning-"


def init_dagshub():
    """Connect to DagsHub MLflow server."""
    print("Connecting to DagsHub MLflow server...")
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    print("Connected.\n")


def get_existing_run_names():
    """Query MLflow server and return set of run names already logged."""
    try:
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        existing = set()
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
            for run in runs:
                name = run.data.tags.get("mlflow.runName")
                # Only count runs that actually have artifacts (fully logged)
                if name and run.info.status == "FINISHED":
                    existing.add(name)
        print(f"Found {len(existing)} existing runs on DagsHub: {sorted(existing)}\n")
        return existing
    except Exception as e:
        print(f"Warning: could not fetch existing runs: {e}")
        return set()


def log_baseline_run(model_dir: Path):
    """Log a single Ultralytics baseline training run."""
    model_name = model_dir.name.replace("_baseline", "")
    run_name = f"{model_name}_baseline"

    print(f"\n{'='*60}")
    print(f"Logging: {run_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("architecture", "baseline")
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("dataset", "DiaMOS Plant")
        mlflow.set_tag("task", "detection_only")

        # Params from args.yaml
        args_path = model_dir / "args.yaml"
        if args_path.exists():
            with open(args_path, "r") as f:
                args = yaml.safe_load(f)
            for k in ["epochs", "batch", "imgsz", "lr0", "lrf", "momentum",
                      "weight_decay", "warmup_epochs", "patience", "optimizer",
                      "mosaic", "copy_paste", "mixup", "fliplr",
                      "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
                      "scale", "shear"]:
                if k in args:
                    mlflow.log_param(k, args[k])
        mlflow.log_param("model", model_name)

        # Final metrics from metrics_summary.json
        summary_path = model_dir / "metrics_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                metrics = json.load(f)
            for k, v in metrics.items():
                safe_log_metric(k, v)

        # Per-epoch metrics from results.csv
        results_csv = model_dir / "results.csv"
        if results_csv.exists():
            with open(results_csv, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    for k, v in row.items():
                        if k and k.strip():
                            safe_log_metric(k, v, step=i)
            print(f"  Logged per-epoch metrics from {results_csv.name}")

        # Artifacts
        best_pt = model_dir / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            print(f"  Logged best.pt ({best_pt.stat().st_size / 1e6:.1f} MB)")

        for img_name in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                          "results.png", "BoxF1_curve.png", "BoxPR_curve.png",
                          "BoxP_curve.png", "BoxR_curve.png", "labels.jpg"]:
            img = model_dir / img_name
            if img.exists():
                mlflow.log_artifact(str(img), artifact_path="plots")

        print(f"  DONE: {run_name}")


def log_multitask_run(model_dir: Path, run_label: str = None):
    """Log a multi-task training run."""
    run_name = run_label or model_dir.name

    print(f"\n{'='*60}")
    print(f"Logging: {run_name}")
    print(f"{'='*60}")

    history_path = model_dir / "training_history.json"
    if not history_path.exists():
        print(f"  SKIP: no training_history.json found")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    train_hist = history.get("train", [])
    val_hist = history.get("val", [])
    n_epochs = len(train_hist)
    if n_epochs == 0:
        print(f"  SKIP: empty history")
        return

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("architecture", "CBAM-YOLO-MT")
        mlflow.set_tag("model_family", "yolo11n")
        mlflow.set_tag("dataset", "DiaMOS Plant")
        mlflow.set_tag("task", "detection + severity")

        # Hyperparameters
        mlflow.log_param("model", "yolo11n")
        mlflow.log_param("epochs", EPOCHS_MULTITASK)
        mlflow.log_param("epochs_completed", n_epochs)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("imgsz", IMG_SIZE)
        mlflow.log_param("lr", LEARNING_RATE)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("lambda_severity", LAMBDA_SEVERITY)
        mlflow.log_param("use_cbam", USE_CBAM)
        mlflow.log_param("use_bifpn", USE_BIFPN)
        mlflow.log_param("use_severity_gate", USE_SEVERITY_GATE)
        mlflow.log_param("optimizer", "SGD")

        # Per-epoch metrics
        for i, (train_m, val_m) in enumerate(zip(train_hist, val_hist)):
            for k, v in train_m.items():
                safe_log_metric(f"train_{k}", v, step=i)
            for k, v in val_m.items():
                safe_log_metric(f"val_{k}", v, step=i)

        # Final best metrics
        best_val = min(val_hist, key=lambda x: x["total_loss"])
        best_epoch = val_hist.index(best_val) + 1
        safe_log_metric("best_epoch", best_epoch)
        safe_log_metric("best_val_total_loss", best_val["total_loss"])
        safe_log_metric("best_val_det_loss", best_val["det_loss"])
        safe_log_metric("best_val_sev_loss", best_val["sev_loss"])
        safe_log_metric("best_val_sev_acc", best_val.get("sev_acc", 0))
        safe_log_metric("best_val_sev_mae", best_val.get("sev_mae", 0))

        # Artifacts
        best_pt = model_dir / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            print(f"  Logged best.pt ({best_pt.stat().st_size / 1e6:.1f} MB)")

        mlflow.log_artifact(str(history_path), artifact_path="history")
        print(f"  DONE: {run_name} ({n_epochs} epochs, best epoch {best_epoch})")


def log_comparison_table():
    """Log the final comparison table as a shared artifact."""
    print(f"\n{'='*60}")
    print("Logging comparison table")
    print(f"{'='*60}")

    with mlflow.start_run(run_name="FINAL_comparison"):
        mlflow.set_tag("type", "summary")

        csv_path = TABLES_DIR / "comparison_table.csv"
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), artifact_path="tables")
            print(f"  Logged comparison_table.csv")

        json_path = TABLES_DIR / "detailed_metrics.json"
        if json_path.exists():
            mlflow.log_artifact(str(json_path), artifact_path="tables")
            print(f"  Logged detailed_metrics.json")

        readme = PROJECT_ROOT / "README.md"
        if readme.exists():
            mlflow.log_artifact(str(readme), artifact_path="docs")
            print(f"  Logged README.md")


def main():
    init_dagshub()
    existing_runs = get_existing_run_names()

    # 1. Baselines
    baselines_dir = MODELS_DIR / "baselines"
    if baselines_dir.exists():
        for model_dir in sorted(baselines_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            run_name = f"{model_dir.name.replace('_baseline', '')}_baseline"
            if run_name in existing_runs:
                print(f"SKIP (already logged): {run_name}")
                continue
            try:
                log_baseline_run(model_dir)
            except Exception as e:
                print(f"  ERROR logging {model_dir.name}: {e}")

    # 2. Multi-task runs — log ALL folders that have a valid training_history.json
    multitask_dir = MODELS_DIR / "multitask"
    if multitask_dir.exists():
        for model_dir in sorted(multitask_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            hist_path = model_dir / "training_history.json"
            if not hist_path.exists():
                continue  # silently skip empty/failed runs
            try:
                with open(hist_path) as f:
                    h = json.load(f)
                n = len(h.get("train", []))
                if n == 0:
                    continue

                folder_tag = model_dir.name.replace("yolo11n_multitask_", "").replace("lambda0.5_", "")
                run_label = f"cbam_yolo_mt_{folder_tag}_{n}ep"
                if run_label in existing_runs:
                    print(f"SKIP (already logged): {run_label}")
                    continue
                log_multitask_run(model_dir, run_label=run_label)
            except Exception as e:
                print(f"  ERROR logging {model_dir.name}: {e}")

    # 3. Comparison summary
    if "FINAL_comparison" in existing_runs:
        print(f"SKIP (already logged): FINAL_comparison")
    else:
        try:
            log_comparison_table()
        except Exception as e:
            print(f"  ERROR logging comparison: {e}")

    print("\n" + "=" * 60)
    print("ALL RUNS LOGGED TO DAGSHUB MLFLOW")
    print(f"Dashboard: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
