"""
Task 1: Dataset Exploration
- Load all YOLO annotations and CSV severity labels
- Map each image to its bounding box + severity
- Report class distributions and statistics
- Visualize sample images with bounding boxes
- Check for missing annotations / mismatches
"""
import sys
import json
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DATA_ROOT, LEAVES_DIR, YOLO_ANNOT_DIR, CSV_PATH,
    FIGURES_DIR, REPORTS_DIR, CLASS_NAMES, SUBDIR_TO_CLASS_IDX,
    CSV_COL_TO_CLASS, SEVERITY_NAMES, SEVERITY_NOT_ESTIMABLE,
)

logger = logging.getLogger(__name__)


def load_csv_annotations(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the diaMOSPlant.csv and parse disease class + severity."""
    logger.info(f"Loading CSV annotations from {csv_path}")
    df = pd.read_csv(csv_path, sep=";")

    disease_col_map = {
        "healthy": "healthy",
        "pear_slug": "slug",
        "leaf_spot": "spot",
        "curl": "curl",
    }
    def get_disease(row):
        for col, name in disease_col_map.items():
            if row[col] == 1:
                return name
        return "unknown"

    df["disease_class"] = df.apply(get_disease, axis=1)

    sev_cols = ["severity_0", "severity_1", "severity_2", "severity_3", "severity_4"]
    def get_severity(row):
        for i, col in enumerate(sev_cols):
            val = row[col]
            # Handle mixed types: pandas reads column as object when
            # "not estimable" strings are mixed with 0/1 integers
            if isinstance(val, str):
                if "not estimable" in val.lower():
                    return SEVERITY_NOT_ESTIMABLE
                try:
                    val = int(val)
                except ValueError:
                    continue
            if val == 1:
                return i
        return SEVERITY_NOT_ESTIMABLE

    df["severity"] = df.apply(get_severity, axis=1).astype(int)
    df["stem"] = df["filename"].apply(lambda x: Path(x).stem)

    logger.info(f"CSV loaded: {len(df)} entries")
    return df


def scan_image_files(leaves_dir: Path = LEAVES_DIR) -> pd.DataFrame:
    """Scan all leaf image files and record their class from subdirectory."""
    logger.info(f"Scanning image files from {leaves_dir}")
    records = []
    for subdir, class_idx in SUBDIR_TO_CLASS_IDX.items():
        class_dir = leaves_dir / subdir
        if not class_dir.exists():
            logger.warning(f"Directory not found: {class_dir}")
            continue
        for img_path in sorted(class_dir.glob("*.*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                records.append({
                    "filename": img_path.name,
                    "stem": img_path.stem,
                    "subdir": subdir,
                    "class_idx": class_idx,
                    "class_name": CLASS_NAMES[class_idx],
                    "image_path": str(img_path),
                })
    logger.info(f"Found {len(records)} image files")
    return pd.DataFrame(records)


def scan_yolo_annotations(annot_dir: Path = YOLO_ANNOT_DIR) -> pd.DataFrame:
    """Read all YOLO .txt annotation files."""
    logger.info(f"Scanning YOLO annotations from {annot_dir}")
    records = []
    for txt_path in sorted(annot_dir.glob("*.txt")):
        stem = txt_path.stem
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    boxes.append({
                        "orig_class_id": cls_id,
                        "x_center": x_c,
                        "y_center": y_c,
                        "width": w,
                        "height": h,
                    })
        records.append({
            "stem": stem,
            "annot_path": str(txt_path),
            "num_boxes": len(boxes),
            "boxes": boxes,
        })
    logger.info(f"Found {len(records)} YOLO annotation files")
    return pd.DataFrame(records)


def build_unified_dataset():
    """Merge image files, YOLO annotations, and CSV severity into one DataFrame."""
    logger.info("=" * 60)
    logger.info("LOADING DATA SOURCES")

    img_df = scan_image_files()
    annot_df = scan_yolo_annotations()
    csv_df = load_csv_annotations()

    merged = img_df.merge(annot_df, on="stem", how="outer", indicator="_merge_yolo")
    csv_subset = csv_df[["stem", "disease_class", "severity"]].rename(
        columns={"disease_class": "csv_disease", "severity": "csv_severity"}
    )
    merged = merged.merge(csv_subset, on="stem", how="outer", indicator="_merge_csv")

    logger.info(f"Unified dataset built: {len(merged)} rows")
    return merged, img_df, annot_df, csv_df


def report_statistics(merged, img_df, annot_df, csv_df):
    """Log comprehensive dataset statistics."""
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")

    # Class distribution from filesystem
    logger.info("--- Image Count by Class (filesystem) ---")
    class_counts = img_df["class_name"].value_counts()
    total = class_counts.sum()
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        pct = 100 * count / total
        logger.info(f"  {cls:>10s}: {count:>5d}  ({pct:5.1f}%)")
    logger.info(f"  {'TOTAL':>10s}: {total:>5d}")

    max_cls = class_counts.max()
    min_cls = class_counts.min()
    logger.info(f"Imbalance ratio (max/min): {max_cls/min_cls:.1f}x")

    # Severity distribution from CSV
    logger.info("--- Severity Distribution (CSV) ---")
    sev_counts = csv_df["severity"].value_counts().sort_index()
    for sev, count in sev_counts.items():
        label = SEVERITY_NAMES[sev] if sev >= 0 else "not_estimable"
        logger.info(f"  Severity {sev:>2d} ({label:>15s}): {count:>5d}")

    # Cross-tabulation: class x severity
    valid_csv = csv_df[csv_df["severity"] >= 0].copy()
    if not valid_csv.empty:
        ct = pd.crosstab(valid_csv["disease_class"], valid_csv["severity"], margins=True)
        logger.info(f"Class x Severity Cross-Tabulation:\n{ct.to_string()}")

    # Annotation matching
    has_img = merged["image_path"].notna()
    has_yolo = merged["annot_path"].notna()
    has_csv = merged["csv_severity"].notna()

    logger.info("--- Annotation Matching Report ---")
    logger.info(f"Images with YOLO annotation:  {(has_img & has_yolo).sum()} / {has_img.sum()}")
    logger.info(f"Images with CSV severity:     {(has_img & has_csv).sum()} / {has_img.sum()}")
    logger.info(f"Images with both:             {(has_img & has_yolo & has_csv).sum()} / {has_img.sum()}")

    img_only = merged[has_img & ~has_yolo]
    yolo_only = merged[~has_img & has_yolo]
    csv_only = merged[has_img & ~has_csv]

    if len(img_only) > 0:
        logger.warning(f"{len(img_only)} images WITHOUT YOLO annotation:")
        for _, r in img_only.head(10).iterrows():
            logger.warning(f"  {r['stem']} ({r.get('class_name', '?')})")

    if len(yolo_only) > 0:
        logger.warning(f"{len(yolo_only)} YOLO annotations WITHOUT image:")
        for _, r in yolo_only.head(10).iterrows():
            logger.warning(f"  {r['stem']}")

    if len(csv_only) > 0:
        logger.info(f"{len(csv_only)} images WITHOUT CSV severity entry:")
        for _, r in csv_only.head(10).iterrows():
            logger.info(f"  {r['stem']} ({r.get('class_name', '?')})")

    # YOLO annotation stats
    box_counts = annot_df["num_boxes"].describe()
    logger.info(f"YOLO BBox stats: min={int(box_counts['min'])}, "
                f"max={int(box_counts['max'])}, mean={box_counts['mean']:.2f}")
    logger.info(f"Images with exactly 1 box: {(annot_df['num_boxes'] == 1).sum()} / {len(annot_df)}")

    all_cls_ids = []
    for _, row in annot_df.iterrows():
        for box in row["boxes"]:
            all_cls_ids.append(box["orig_class_id"])
    unique_ids = set(all_cls_ids)
    logger.info(f"Unique original class IDs in YOLO: {unique_ids}")

    return class_counts, sev_counts


def plot_class_distribution(class_counts, save_dir: Path = FIGURES_DIR):
    """Bar chart of class distribution."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
    bars = axes[0].bar(class_counts.index, class_counts.values, color=colors)
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_xlabel("Disease Class")
    for bar, v in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     str(v), ha="center", fontweight="bold")

    axes[1].pie(class_counts.values, labels=class_counts.index,
                autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title("Class Proportions", fontsize=14, fontweight="bold")

    plt.tight_layout()
    out_path = save_dir / "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_severity_distribution(csv_df, save_dir: Path = FIGURES_DIR):
    """Severity distribution and class-severity heatmap."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid = csv_df[csv_df["severity"] >= 0]
    sev_counts = valid["severity"].value_counts().sort_index()
    sev_labels = [SEVERITY_NAMES[i] for i in sev_counts.index]
    colors_sev = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    axes[0].bar(sev_labels, sev_counts.values, color=colors_sev)
    axes[0].set_title("Severity Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(sev_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    if len(valid) > 0:
        ct = pd.crosstab(valid["disease_class"], valid["severity"])
        ct.columns = [SEVERITY_NAMES[int(c)] for c in ct.columns]
        sns.heatmap(ct.astype(int), annot=True, fmt="d", cmap="YlOrRd", ax=axes[1])
        axes[1].set_title("Class x Severity Heatmap", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Disease Class")
        axes[1].set_xlabel("Severity Level")
    else:
        axes[1].text(0.5, 0.5, "No valid severity data", ha="center", va="center")
        axes[1].set_title("Class x Severity Heatmap")
        logger.warning("No valid severity data for heatmap")

    plt.tight_layout()
    out_path = save_dir / "severity_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_sample_images_with_bboxes(img_df, annot_df, n_per_class=2,
                                    save_dir: Path = FIGURES_DIR):
    """Draw bounding boxes on sample images from each class."""
    save_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, n_per_class,
                              figsize=(5 * n_per_class, 5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    colors_bgr = [
        (46, 204, 113), (231, 76, 60), (243, 156, 18), (52, 152, 219),
    ]

    for row_idx, cls_name in enumerate(CLASS_NAMES):
        cls_imgs = img_df[img_df["class_name"] == cls_name].sample(
            n=min(n_per_class, len(img_df[img_df["class_name"] == cls_name])),
            random_state=42,
        )
        for col_idx, (_, img_row) in enumerate(cls_imgs.iterrows()):
            ax = axes[row_idx, col_idx]
            img = cv2.imread(img_row["image_path"])
            if img is None:
                logger.error(f"Failed to load image: {img_row['image_path']}")
                ax.set_title(f"{cls_name} - LOAD ERROR")
                ax.axis("off")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img = img.shape[:2]

            annot_row = annot_df[annot_df["stem"] == img_row["stem"]]
            if not annot_row.empty:
                boxes = annot_row.iloc[0]["boxes"]
                for box in boxes:
                    xc = box["x_center"] * w_img
                    yc = box["y_center"] * h_img
                    bw = box["width"] * w_img
                    bh = box["height"] * h_img
                    x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
                    x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
                    color = colors_bgr[row_idx]
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img_rgb, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ax.imshow(img_rgb)
            ax.set_title(f"{cls_name} | {img_row['filename']} | {h_img}x{w_img}", fontsize=10)
            ax.axis("off")

    plt.suptitle("Sample Images with YOLO Bounding Boxes",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = save_dir / "sample_images_with_bbox.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_image_dimensions(img_df, save_dir: Path = FIGURES_DIR):
    """Histogram of image dimensions."""
    save_dir.mkdir(parents=True, exist_ok=True)
    widths, heights = [], []
    for _, row in img_df.sample(n=min(500, len(img_df)), random_state=42).iterrows():
        img = cv2.imread(row["image_path"])
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(widths, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[1].hist(heights, bins=30, color="#e74c3c", edgecolor="black", alpha=0.7)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")
    plt.tight_layout()
    out_path = save_dir / "image_dimensions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def save_exploration_report(merged, class_counts, sev_counts,
                            save_dir: Path = REPORTS_DIR):
    """Save a text summary report."""
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "exploration_report.txt"

    with open(report_path, "w") as f:
        f.write("DiaMOS Plant Dataset - Exploration Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("CLASS DISTRIBUTION:\n")
        for cls in CLASS_NAMES:
            count = class_counts.get(cls, 0)
            f.write(f"  {cls}: {count}\n")
        f.write(f"  Total: {class_counts.sum()}\n\n")
        f.write("SEVERITY DISTRIBUTION:\n")
        for sev, count in sev_counts.items():
            label = SEVERITY_NAMES[sev] if sev >= 0 else "not_estimable"
            f.write(f"  {label}: {count}\n")
        f.write("\n")
        total_imgs = merged["image_path"].notna().sum()
        has_yolo = (merged["image_path"].notna() & merged["annot_path"].notna()).sum()
        has_csv = (merged["image_path"].notna() & merged["csv_severity"].notna()).sum()
        f.write(f"ANNOTATION COVERAGE:\n")
        f.write(f"  Total images: {total_imgs}\n")
        f.write(f"  With YOLO bbox: {has_yolo}\n")
        f.write(f"  With CSV severity: {has_csv}\n")

    logger.info(f"Exploration report saved: {report_path}")


def run_exploration():
    """Run the full dataset exploration pipeline."""
    logger.info("#" * 60)
    logger.info("TASK 1: DATASET EXPLORATION")
    logger.info("#" * 60)

    merged, img_df, annot_df, csv_df = build_unified_dataset()
    class_counts, sev_counts = report_statistics(merged, img_df, annot_df, csv_df)

    logger.info("--- Generating Visualizations ---")
    plot_class_distribution(class_counts)
    plot_severity_distribution(csv_df)
    plot_sample_images_with_bboxes(img_df, annot_df)
    plot_image_dimensions(img_df)

    save_exploration_report(merged, class_counts, sev_counts)

    mapping = []
    valid_merged = merged[merged["image_path"].notna()].copy()
    for _, row in valid_merged.iterrows():
        entry = {
            "stem": row["stem"],
            "filename": row.get("filename", ""),
            "image_path": row.get("image_path", ""),
            "class_name": row.get("class_name", "unknown"),
            "class_idx": int(row.get("class_idx", -1)),
            "annot_path": row.get("annot_path", ""),
            "severity": int(row["csv_severity"]) if pd.notna(row.get("csv_severity")) else SEVERITY_NOT_ESTIMABLE,
        }
        mapping.append(entry)

    mapping_path = REPORTS_DIR / "dataset_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Dataset mapping saved: {mapping_path}")

    logger.info("=" * 60)
    logger.info("EXPLORATION COMPLETE")
    return merged, img_df, annot_df, csv_df


if __name__ == "__main__":
    run_exploration()
