"""
Task 7: Paper-Ready Outputs
- LaTeX tables for results
- Confusion matrix for severity
- Training curves (loss, mAP vs epochs)
- Per-class AP bar chart
- Sample detection visualizations
"""
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR,
    CLASS_NAMES, SEVERITY_NAMES, NUM_SEVERITY,
)

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.labelsize": 13,
    "axes.titlesize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 150,
})


def generate_latex_comparison_table(csv_path=None):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path is None:
        csv_path = TABLES_DIR / "comparison_table.csv"
    if not csv_path.exists():
        logger.warning(f"No comparison table found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    cols = ["model", "mAP50", "precision", "recall", "severity_acc"]
    available_cols = [c for c in cols if c in df.columns]
    sub_df = df[available_cols].copy()

    header_map = {"model": "Model", "mAP50": "mAP@0.5", "precision": "Precision",
                  "recall": "Recall", "severity_acc": "Sev. Acc."}

    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += "\\caption{Detection and severity prediction results on DiaMOS Plant test set.}\n"
    latex += "\\label{tab:comparison}\n"
    latex += "\\begin{tabular}{l" + "c" * (len(available_cols) - 1) + "}\n\\toprule\n"
    latex += " & ".join([header_map.get(c, c) for c in available_cols]) + " \\\\\n\\midrule\n"
    for _, row in sub_df.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) and not np.isnan(row[c]) else str(row[c]) for c in available_cols]
        latex += " & ".join(vals) + " \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    tex_path = TABLES_DIR / "comparison_table.tex"
    tex_path.write_text(latex)
    logger.info(f"LaTeX comparison table saved: {tex_path}")

    ap_cols = [c for c in df.columns if c.startswith("AP50_")]
    if ap_cols:
        latex_ap = "\\begin{table}[htbp]\n\\centering\n"
        latex_ap += "\\caption{Per-class AP@0.5 on DiaMOS Plant test set.}\n\\label{tab:per_class_ap}\n"
        latex_ap += "\\begin{tabular}{l" + "c" * len(ap_cols) + "}\n\\toprule\n"
        latex_ap += " & ".join(["Model"] + [c.replace("AP50_", "") for c in ap_cols]) + " \\\\\n\\midrule\n"
        for _, row in df.iterrows():
            vals = [str(row["model"])] + [f"{row[c]:.4f}" if isinstance(row[c], float) and not np.isnan(row[c]) else "--" for c in ap_cols]
            latex_ap += " & ".join(vals) + " \\\\\n"
        latex_ap += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        tex_ap_path = TABLES_DIR / "per_class_ap_table.tex"
        tex_ap_path.write_text(latex_ap)
        logger.info(f"LaTeX per-class AP table saved: {tex_ap_path}")


def generate_latex_ablation_tables():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    lambda_csv = TABLES_DIR / "ablation_lambda.csv"
    if lambda_csv.exists():
        df = pd.read_csv(lambda_csv)
        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += "\\caption{Ablation study: effect of severity loss weight $\\lambda$.}\n\\label{tab:ablation_lambda}\n"
        latex += "\\begin{tabular}{cccccc}\n\\toprule\n"
        latex += "$\\lambda$ & Total Loss & Det. Loss & Sev. Loss & Sev. Acc. & Sev. MAE \\\\\n\\midrule\n"
        for _, row in df.iterrows():
            latex += (f"{row['lambda']:.1f} & {row['best_val_loss']:.4f} & {row['best_det_loss']:.4f} & "
                      f"{row['best_sev_loss']:.4f} & {row['best_sev_acc']:.4f} & {row['best_sev_mae']:.4f} \\\\\n")
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        (TABLES_DIR / "ablation_lambda.tex").write_text(latex)
        logger.info(f"LaTeX lambda ablation table saved")

    aug_csv = TABLES_DIR / "ablation_augmentation.csv"
    if aug_csv.exists():
        df = pd.read_csv(aug_csv)
        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += "\\caption{Ablation study: effect of minority class oversampling.}\n\\label{tab:ablation_augmentation}\n"
        latex += "\\begin{tabular}{lccc}\n\\toprule\n"
        latex += "Configuration & mAP@0.5 & Precision & Recall \\\\\n\\midrule\n"
        for _, row in df.iterrows():
            latex += (f"{row['config'].replace('_', ' ').title()} & {row['mAP50']:.4f} & "
                      f"{row['precision']:.4f} & {row['recall']:.4f} \\\\\n")
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        (TABLES_DIR / "ablation_augmentation.tex").write_text(latex)
        logger.info(f"LaTeX augmentation ablation table saved")


def plot_training_curves(history_path=None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if history_path is None:
        multitask_dir = MODELS_DIR / "multitask"
        if not multitask_dir.exists():
            logger.warning("No multi-task training history found")
            return
        for run_dir in reversed(sorted(multitask_dir.iterdir())):
            hp = run_dir / "training_history.json"
            if hp.exists():
                history_path = hp
                break
    if history_path is None or not Path(history_path).exists():
        logger.warning("No training history file found")
        return

    with open(history_path, "r") as f:
        history = json.load(f)
    train_hist, val_hist = history["train"], history["val"]
    epochs = range(1, len(train_hist) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key, title in [
        (axes[0,0], "total_loss", "Total Loss"), (axes[0,1], "det_loss", "Detection Loss"),
        (axes[1,0], "sev_loss", "Severity Loss"),
    ]:
        ax.plot(epochs, [h[key] for h in train_hist], "b-", label="Train", linewidth=2)
        ax.plot(epochs, [h[key] for h in val_hist], "r--", label="Val", linewidth=2)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    axes[1,1].plot(epochs, [h.get("sev_acc", 0) for h in train_hist], "b-", label="Train", linewidth=2)
    axes[1,1].plot(epochs, [h.get("sev_acc", 0) for h in val_hist], "r--", label="Val", linewidth=2)
    axes[1,1].set_title("Severity Accuracy"); axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Accuracy")
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3); axes[1,1].set_ylim(0, 1)

    plt.suptitle("Multi-Task YOLOv11 Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


def plot_severity_confusion_matrix(metrics_path=None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if metrics_path is None:
        metrics_path = TABLES_DIR / "detailed_metrics.json"
    if not Path(metrics_path).exists():
        logger.warning("No detailed metrics found for confusion matrix")
        return
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    cm = None
    for m in all_metrics:
        if "severity_confusion_matrix" in m:
            cm = np.array(m["severity_confusion_matrix"]); break
    if cm is None:
        logger.warning("No severity confusion matrix found in metrics")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=SEVERITY_NAMES, yticklabels=SEVERITY_NAMES, ax=ax)
    ax.set_title("Severity Prediction Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Severity"); ax.set_ylabel("True Severity")
    plt.tight_layout()
    out = FIGURES_DIR / "severity_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


def plot_per_class_ap_chart():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "comparison_table.csv"
    if not csv_path.exists():
        logger.warning("No comparison table found for per-class AP chart")
        return
    df = pd.read_csv(csv_path)
    ap_cols = [c for c in df.columns if c.startswith("AP50_")]
    if not ap_cols:
        logger.warning("No per-class AP data available"); return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ap_cols))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (_, row) in enumerate(df.iterrows()):
        values = [float(row[c]) if pd.notna(row[c]) and row[c] != "--" else 0 for c in ap_cols]
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=row["model"], color=colors[i % len(colors)], edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("AP50_", "") for c in ap_cols])
    ax.set_ylabel("AP@0.5"); ax.set_title("Per-Class AP@0.5 Comparison", fontsize=14, fontweight="bold")
    ax.legend(); ax.set_ylim(0, 1.05); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "per_class_ap_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


def plot_lambda_ablation():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "ablation_lambda.csv"
    if not csv_path.exists():
        logger.warning("No lambda ablation data found"); return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(df["lambda"], df["best_det_loss"], "bo-", label="Detection Loss", linewidth=2, markersize=8)
    axes[0].plot(df["lambda"], df["best_sev_loss"], "rs-", label="Severity Loss", linewidth=2, markersize=8)
    axes[0].set_xlabel("$\\lambda$"); axes[0].set_ylabel("Val Loss"); axes[0].set_title("Loss vs $\\lambda$"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["lambda"], df["best_sev_acc"], "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("$\\lambda$"); axes[1].set_ylabel("Sev. Accuracy"); axes[1].set_title("Sev. Accuracy vs $\\lambda$"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["lambda"], df["best_val_loss"], "mo-", linewidth=2, markersize=8)
    axes[2].set_xlabel("$\\lambda$"); axes[2].set_ylabel("Total Val Loss"); axes[2].set_title("Total Loss vs $\\lambda$"); axes[2].grid(True, alpha=0.3)
    plt.suptitle("Ablation: Effect of $\\lambda$", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "ablation_lambda.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


def plot_sample_detections(n_samples=8):
    import cv2
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    baselines_dir = MODELS_DIR / "baselines"
    best_pt = None
    for model_dir in sorted(baselines_dir.iterdir()) if baselines_dir.exists() else []:
        candidate = model_dir / "weights" / "best.pt"
        if candidate.exists(): best_pt = candidate; break
    if best_pt is None:
        logger.warning("No trained model found for detection visualization"); return

    from ultralytics import YOLO
    from src.config import PROCESSED_DIR
    model = YOLO(str(best_pt))
    test_images = sorted((PROCESSED_DIR / "images" / "test").glob("*.jpg"))[:n_samples]
    if not test_images:
        logger.warning("No test images found"); return

    n_cols = 4
    n_rows = (len(test_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    colors = {"healthy": (46,204,113), "spot": (231,76,60), "curl": (243,156,18), "slug": (52,152,219)}

    for i, img_path in enumerate(test_images):
        if i >= len(axes): break
        results = model.predict(str(img_path), imgsz=640, conf=0.25, verbose=False)
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "?"
                color = colors.get(cls_name, (255,255,255))
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
                cv2.putText(img, f"{cls_name} {conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        axes[i].imshow(img); axes[i].set_title(img_path.stem, fontsize=10); axes[i].axis("off")
    for j in range(len(test_images), len(axes)): axes[j].axis("off")

    plt.suptitle("Sample Detection Results on Test Set", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "sample_detections.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


def run_visualization():
    logger.info("#" * 60)
    logger.info("TASK 7: PAPER-READY OUTPUTS")
    logger.info("#" * 60)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[1/6] Generating LaTeX comparison table...")
    generate_latex_comparison_table()
    logger.info("[2/6] Generating LaTeX ablation tables...")
    generate_latex_ablation_tables()
    logger.info("[3/6] Plotting training curves...")
    plot_training_curves()
    logger.info("[4/6] Plotting severity confusion matrix...")
    plot_severity_confusion_matrix()
    logger.info("[5/6] Plotting per-class AP chart...")
    plot_per_class_ap_chart()
    logger.info("[6/6] Plotting sample detections...")
    plot_sample_detections()
    logger.info("Generating lambda ablation plot...")
    plot_lambda_ablation()

    logger.info("PAPER-READY OUTPUTS COMPLETE")


if __name__ == "__main__":
    run_visualization()
