"""
Task 2: Data Pipeline
- Remap YOLO class IDs from 0 -> actual disease class
- Stratified train/val/test split (70:20:10)
- Create YOLO-format directory structure for Ultralytics
- Generate data.yaml
- Create severity label mapping (JSON)
- Handle class imbalance via oversampling minority classes
"""
import sys
import json
import logging
import shutil
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    LEAVES_DIR, YOLO_ANNOT_DIR, CSV_PATH, PROCESSED_DIR,
    CLASS_NAMES, SUBDIR_TO_CLASS_IDX, NUM_CLASSES, NUM_SEVERITY,
    SEVERITY_NOT_ESTIMABLE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, IMG_SIZE,
)
from src.data.explore import load_csv_annotations, scan_image_files, scan_yolo_annotations

logger = logging.getLogger(__name__)


def build_image_annotation_list():
    """Build a list of (image_path, annot_path, class_idx, severity) tuples."""
    logger.info("Building image-annotation mapping...")
    img_df = scan_image_files()
    annot_df = scan_yolo_annotations()
    csv_df = load_csv_annotations()

    severity_map = dict(zip(csv_df["stem"], csv_df["severity"]))
    annot_map = {}
    for _, row in annot_df.iterrows():
        annot_map[row["stem"]] = {
            "annot_path": row["annot_path"],
            "boxes": row["boxes"],
        }

    records = []
    skipped = 0
    for _, img_row in img_df.iterrows():
        stem = img_row["stem"]
        class_idx = img_row["class_idx"]
        severity = severity_map.get(stem, SEVERITY_NOT_ESTIMABLE)
        annot_info = annot_map.get(stem)
        if annot_info is None:
            skipped += 1
            continue

        records.append({
            "stem": stem,
            "image_path": img_row["image_path"],
            "annot_path": annot_info["annot_path"],
            "boxes": annot_info["boxes"],
            "class_name": img_row["class_name"],
            "class_idx": class_idx,
            "severity": severity,
        })

    if skipped > 0:
        logger.warning(f"Skipped {skipped} images without YOLO annotation")
    logger.info(f"Total images with annotations: {len(records)}")
    return records


def remap_yolo_annotation(boxes, new_class_id):
    """Rewrite YOLO annotation with the correct class ID."""
    lines = []
    for box in boxes:
        lines.append(
            f"{new_class_id} {box['x_center']:.6f} {box['y_center']:.6f} "
            f"{box['width']:.6f} {box['height']:.6f}"
        )
    return "\n".join(lines)


def stratified_split(records, seed=RANDOM_SEED):
    """Stratified train/val/test split preserving class proportions."""
    logger.info("Performing stratified train/val/test split...")
    df = pd.DataFrame(records)

    train_val_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=seed, stratify=df["class_idx"],
    )
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_fraction, random_state=seed,
        stratify=train_val_df["class_idx"],
    )

    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return (
        train_df.to_dict("records"),
        val_df.to_dict("records"),
        test_df.to_dict("records"),
    )


def oversample_minority_classes(records, target_ratio=0.5):
    """Oversample minority classes (healthy, curl) by duplicating samples."""
    class_counts = Counter(r["class_idx"] for r in records)
    max_count = max(class_counts.values())
    target_min = int(max_count * target_ratio)

    oversampled = list(records)
    for cls_idx, count in class_counts.items():
        if count < target_min:
            cls_records = [r for r in records if r["class_idx"] == cls_idx]
            n_extra = target_min - count
            extra = [cls_records[i % len(cls_records)] for i in range(n_extra)]
            for i, rec in enumerate(extra):
                new_rec = dict(rec)
                new_rec["_oversample_idx"] = i
                oversampled.append(new_rec)
            logger.info(f"Oversampled class {cls_idx} ({CLASS_NAMES[cls_idx]}): {count} -> {count + n_extra}")

    return oversampled


def write_split_to_disk(records, split_name, output_dir, oversample_train=True):
    """Write images and remapped YOLO labels to the YOLO-format directory."""
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    severity_map = {}
    written = 0

    for rec in records:
        stem = rec["stem"]
        oversample_idx = rec.get("_oversample_idx")
        new_stem = f"{stem}_aug{oversample_idx}" if oversample_idx is not None else stem

        src_img = Path(rec["image_path"])
        dst_img = img_dir / f"{new_stem}{src_img.suffix}"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        label_content = remap_yolo_annotation(rec["boxes"], rec["class_idx"])
        dst_lbl = lbl_dir / f"{new_stem}.txt"
        dst_lbl.write_text(label_content)

        severity_map[new_stem] = {
            "severity": rec["severity"],
            "class_idx": rec["class_idx"],
            "class_name": rec["class_name"],
            "original_stem": stem,
        }
        written += 1

    logger.info(f"  {split_name}: {written} files written to {img_dir}")
    return severity_map, written


def generate_data_yaml(output_dir):
    """Generate the data.yaml file for Ultralytics YOLO training."""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info(f"data.yaml saved: {yaml_path}")
    return yaml_path


def run_preparation():
    """Run the full data preparation pipeline."""
    logger.info("#" * 60)
    logger.info("TASK 2: DATA PREPARATION")
    logger.info("#" * 60)

    output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/5] Building image-annotation mapping...")
    records = build_image_annotation_list()

    logger.info("[2/5] Performing stratified train/val/test split...")
    train_recs, val_recs, test_recs = stratified_split(records)

    for name, recs in [("Train", train_recs), ("Val", val_recs), ("Test", test_recs)]:
        counts = Counter(r["class_name"] for r in recs)
        logger.info(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    logger.info("[3/5] Oversampling minority classes in training set...")
    original_train_count = len(train_recs)
    train_recs_oversampled = oversample_minority_classes(train_recs, target_ratio=0.3)
    logger.info(f"Training samples: {original_train_count} -> {len(train_recs_oversampled)} "
                f"(+{len(train_recs_oversampled) - original_train_count} oversampled)")

    logger.info("[4/5] Writing YOLO-format dataset to disk...")
    all_severity = {}
    sev_map, _ = write_split_to_disk(train_recs_oversampled, "train", output_dir)
    all_severity.update(sev_map)
    sev_map, _ = write_split_to_disk(val_recs, "val", output_dir)
    all_severity.update(sev_map)
    sev_map, _ = write_split_to_disk(test_recs, "test", output_dir)
    all_severity.update(sev_map)

    sev_path = output_dir / "severity_labels.json"
    with open(sev_path, "w") as f:
        json.dump(all_severity, f, indent=2)
    logger.info(f"Severity labels saved: {sev_path}")

    logger.info("[5/5] Generating data.yaml...")
    yaml_path = generate_data_yaml(output_dir)

    split_info = []
    for split, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        for r in recs:
            split_info.append({
                "stem": r["stem"], "split": split, "class_name": r["class_name"],
                "class_idx": r["class_idx"], "severity": r["severity"],
            })
    split_df = pd.DataFrame(split_info)
    split_csv_path = output_dir / "split_info.csv"
    split_df.to_csv(split_csv_path, index=False)
    logger.info(f"Split info saved: {split_csv_path}")

    logger.info("=" * 60)
    logger.info("DATA PREPARATION COMPLETE")
    return str(yaml_path)


if __name__ == "__main__":
    run_preparation()
