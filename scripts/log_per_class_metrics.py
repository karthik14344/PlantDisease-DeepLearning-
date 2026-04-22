"""
Log per-class metrics to DagsHub MLflow, matching Mitra 2023's reporting format.

For each BASELINE model:
  - Runs Ultralytics val on the VALIDATION split
  - Logs per-class: precision, recall, f1, mAP@0.5 (exactly like Mitra's Table 7)

For the MULTI-TASK model:
  - Runs detection evaluation on BOTH validation AND test splits
  - Logs per-class: precision, recall, f1, mAP@0.5 (val and test prefixes)
  - Also logs severity accuracy/MAE

Usage:
    python scripts/log_per_class_metrics.py
"""
import sys
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import dagshub
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, CLASS_NAMES, NUM_CLASSES, NUM_SEVERITY,
    IMG_SIZE, BATCH_SIZE, DEVICE,
    USE_CBAM, USE_BIFPN, USE_SEVERITY_GATE,
)

DAGSHUB_OWNER = "karthik14344"
DAGSHUB_REPO = "PlantDisease-DeepLearning-"

# List of (checkpoint_dir, mlflow_run_name) pairs for every multi-task model to log
MULTITASK_RUNS = [
    (
        MODELS_DIR / "multitask" / "yolo11n_multitask_lambda0.5_20260416_0925",
        "cbam_yolo_mt_100ep_per_class_val_test",
    ),
    (
        MODELS_DIR / "multitask" / "_RUN1_cbam_50epochs",
        "cbam_yolo_mt_54ep_per_class_val_test",
    ),
]


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def clean_metric_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[()]", "", name)
    name = name.replace("/", "_").replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-\.:]", "_", name)
    return name


def safe_log_metric(key: str, value):
    try:
        v = float(value)
        if not math.isfinite(v):
            return False
        mlflow.log_metric(clean_metric_name(key), v)
        return True
    except (ValueError, TypeError):
        return False


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def xywhn_to_xyxy(boxes_xywhn, img_w, img_h):
    if len(boxes_xywhn) == 0:
        return torch.zeros((0, 4))
    x_c = boxes_xywhn[:, 0] * img_w
    y_c = boxes_xywhn[:, 1] * img_h
    w = boxes_xywhn[:, 2] * img_w
    h = boxes_xywhn[:, 3] * img_h
    return torch.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], dim=1)


def compute_iou(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_bs = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_bs - inter
    return inter / union.clamp(min=1e-6)


# ---------------------------------------------------------------
# BASELINE: Val-only per-class metrics via Ultralytics
# ---------------------------------------------------------------
def log_baseline_val_metrics(model_dir: Path):
    """Run Ultralytics val on validation split, log per-class metrics."""
    from ultralytics import YOLO

    model_name = model_dir.name.replace("_baseline", "")
    run_name = f"{model_name}_per_class_val"
    best_pt = model_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"SKIP {model_name}: no best.pt")
        return

    print(f"\n{'='*60}\nBaseline: {model_name}  (val split)\n{'='*60}")
    model = YOLO(str(best_pt))
    results = model.val(
        data=str(PROCESSED_DIR / "data.yaml"),
        imgsz=IMG_SIZE, batch=BATCH_SIZE, split="val",
        device="0" if str(DEVICE) == "cuda" else "cpu",
        verbose=False, plots=False,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("type", "per_class_metrics")
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("split", "val")
        mlflow.set_tag("dataset", "DiaMOS Plant")

        # Overall metrics
        rd = results.results_dict
        overall_p = float(rd.get("metrics/precision(B)", 0))
        overall_r = float(rd.get("metrics/recall(B)", 0))
        overall_map50 = float(rd.get("metrics/mAP50(B)", 0))
        overall_map = float(rd.get("metrics/mAP50-95(B)", 0))
        overall_f1 = f1(overall_p, overall_r)

        safe_log_metric("val_overall_precision", overall_p)
        safe_log_metric("val_overall_recall", overall_r)
        safe_log_metric("val_overall_f1", overall_f1)
        safe_log_metric("val_overall_mAP50", overall_map50)
        safe_log_metric("val_overall_mAP50_95", overall_map)

        print(f"  Overall: P={overall_p:.4f} R={overall_r:.4f} F1={overall_f1:.4f} mAP50={overall_map50:.4f}")

        # Per-class metrics from Ultralytics
        try:
            p_arr = results.box.p.tolist()
            r_arr = results.box.r.tolist()
            ap50_arr = results.box.ap50.tolist()
            f1_arr = results.box.f1.tolist()
            for i, cname in enumerate(CLASS_NAMES):
                if i < len(p_arr):
                    p = float(p_arr[i]); r = float(r_arr[i])
                    ap50 = float(ap50_arr[i]); f1_val = float(f1_arr[i])
                    safe_log_metric(f"val_{cname}_precision", p)
                    safe_log_metric(f"val_{cname}_recall", r)
                    safe_log_metric(f"val_{cname}_f1", f1_val)
                    safe_log_metric(f"val_{cname}_mAP50", ap50)
                    print(f"    {cname:>10s}: P={p:.4f} R={r:.4f} F1={f1_val:.4f} mAP50={ap50:.4f}")
        except Exception as e:
            print(f"  Warning: could not extract per-class metrics: {e}")

        mlflow.log_param("model", model_name)
        mlflow.log_param("split", "val")
        mlflow.log_param("imgsz", IMG_SIZE)
        mlflow.log_param("task", "detection_only")
        print(f"  Logged MLflow run: {run_name}")


# ---------------------------------------------------------------
# MULTITASK: Val + Test per-class metrics via custom eval
# ---------------------------------------------------------------
def evaluate_multitask_on_split(model, det_loss_fn, dataset_split, device):
    """Run inference on a split and return per-class P/R/F1 + per-class mAP50 + severity."""
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from src.training.train_multitask import MultiTaskDataset

    with open(PROCESSED_DIR / "severity_labels.json") as f:
        severity_map = json.load(f)

    ds = MultiTaskDataset(
        images_dir=PROCESSED_DIR / "images" / dataset_split,
        labels_dir=PROCESSED_DIR / "labels" / dataset_split,
        severity_map=severity_map, imgsz=IMG_SIZE, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=MultiTaskDataset.collate_fn,
    )
    print(f"  {dataset_split}: {len(ds)} samples")

    map_metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True,
        backend="faster_coco_eval",
    ).to(device)
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    all_sev_preds, all_sev_labels = [], []

    from tqdm import tqdm
    model.eval()
    with torch.no_grad():
        for imgs, batch_labels, severities, _ in tqdm(loader, desc=dataset_split):
            imgs = imgs.to(device)
            batch_labels = batch_labels.to(device)
            severities = severities.to(device)
            _, _, h, w = imgs.shape

            det_preds, sev_logits = model(imgs)
            pred_tensor = det_preds[0] if isinstance(det_preds, tuple) else det_preds
            nms_out = non_max_suppression(pred_tensor, conf_thres=0.001, iou_thres=0.7, max_det=300)

            for bi, det in enumerate(nms_out):
                if det is None or len(det) == 0:
                    pred_boxes = torch.zeros((0, 4), device=device)
                    pred_scores = torch.zeros((0,), device=device)
                    pred_labels = torch.zeros((0,), dtype=torch.long, device=device)
                else:
                    pred_boxes = det[:, :4]; pred_scores = det[:, 4]; pred_labels = det[:, 5].long()

                mask = batch_labels[:, 0] == bi
                gt = batch_labels[mask]
                if len(gt) > 0:
                    gt_boxes = xywhn_to_xyxy(gt[:, 2:6], w, h).to(device)
                    gt_labels = gt[:, 1].long().to(device)
                else:
                    gt_boxes = torch.zeros((0, 4), device=device)
                    gt_labels = torch.zeros((0,), dtype=torch.long, device=device)

                map_metric.update(
                    [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
                    [{"boxes": gt_boxes, "labels": gt_labels}],
                )

                matched_gt = set()
                order = torch.argsort(pred_scores, descending=True)
                for pi in order:
                    pcls = int(pred_labels[pi])
                    gt_idx_same = [i for i, g in enumerate(gt_labels) if int(g) == pcls and i not in matched_gt]
                    if not gt_idx_same:
                        per_class_fp[pcls] += 1
                        continue
                    ious = compute_iou(pred_boxes[pi], gt_boxes[gt_idx_same])
                    best_iou, best = ious.max(0)
                    if best_iou >= 0.5:
                        per_class_tp[pcls] += 1
                        matched_gt.add(gt_idx_same[best.item()])
                    else:
                        per_class_fp[pcls] += 1
                for i, g in enumerate(gt_labels):
                    if i not in matched_gt:
                        per_class_fn[int(g)] += 1

            valid_mask = severities >= 0
            if valid_mask.any():
                sev_preds = sev_logits[valid_mask].argmax(dim=1)
                all_sev_preds.extend(sev_preds.cpu().tolist())
                all_sev_labels.extend(severities[valid_mask].cpu().tolist())

    map_results = map_metric.compute()
    per_class_map = map_results.get("map_per_class", None)

    # Per class metrics
    pc = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp = per_class_tp[i]; fp = per_class_fp[i]; fn = per_class_fn[i]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1_val = f1(p, r)
        ap_val = float(per_class_map[i]) if per_class_map is not None and i < len(per_class_map) else 0.0
        pc[cname] = {"precision": p, "recall": r, "f1": f1_val, "mAP50_95": ap_val,
                     "tp": tp, "fp": fp, "fn": fn}

    total_tp = sum(per_class_tp.values())
    total_fp = sum(per_class_fp.values())
    total_fn = sum(per_class_fn.values())
    overall_p = total_tp / max(total_tp + total_fp, 1)
    overall_r = total_tp / max(total_tp + total_fn, 1)

    all_sev_preds = np.array(all_sev_preds)
    all_sev_labels = np.array(all_sev_labels)
    sev_acc = float((all_sev_preds == all_sev_labels).mean()) if len(all_sev_preds) else 0.0
    sev_mae = float(np.abs(all_sev_preds - all_sev_labels).mean()) if len(all_sev_preds) else 0.0

    return {
        "per_class": pc,
        "overall_precision": overall_p,
        "overall_recall": overall_r,
        "overall_f1": f1(overall_p, overall_r),
        "overall_mAP50": float(map_results["map_50"]),
        "overall_mAP50_95": float(map_results["map"]),
        "severity_acc": sev_acc,
        "severity_mae": sev_mae,
    }


def log_multitask_val_test_metrics(ckpt_dir: Path, run_name: str):
    """Run a CBAM-YOLO-MT checkpoint on val + test splits, log all per-class metrics."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace
    from src.models.multitask_yolo import MultiTaskYOLO

    best_pt = ckpt_dir / "best.pt"
    if not best_pt.exists():
        print(f"SKIP {run_name}: no best.pt at {ckpt_dir}")
        return

    print(f"\n{'='*60}\nMulti-Task: {run_name}\n  ({ckpt_dir})\n{'='*60}")

    device = DEVICE
    yolo = YOLO("yolo11n.pt")
    det_model = DetectionModel(cfg=yolo.model.yaml, nc=NUM_CLASSES, verbose=False)
    det_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    model = MultiTaskYOLO(
        det_model=det_model, num_severity=NUM_SEVERITY, device=str(device),
        use_cbam=USE_CBAM, use_bifpn=USE_BIFPN, use_severity_gate=USE_SEVERITY_GATE,
    ).to(device)

    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    det_loss_fn = v8DetectionLoss(det_model)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("type", "per_class_metrics")
        mlflow.set_tag("model_family", "cbam_yolo_mt")
        mlflow.set_tag("split", "val_and_test")
        mlflow.set_tag("dataset", "DiaMOS Plant")

        mlflow.log_param("model", "CBAM-YOLO-MT (YOLO11n backbone)")
        mlflow.log_param("checkpoint_dir", str(ckpt_dir.name))
        mlflow.log_param("use_cbam", USE_CBAM)
        mlflow.log_param("use_bifpn", USE_BIFPN)
        mlflow.log_param("use_severity_gate", USE_SEVERITY_GATE)
        mlflow.log_param("imgsz", IMG_SIZE)
        mlflow.log_param("epoch_of_best", ckpt.get("epoch"))

        for split in ["val", "test"]:
            print(f"\n  --- Split: {split} ---")
            res = evaluate_multitask_on_split(model, det_loss_fn, split, device)

            # Overall
            safe_log_metric(f"{split}_overall_precision", res["overall_precision"])
            safe_log_metric(f"{split}_overall_recall", res["overall_recall"])
            safe_log_metric(f"{split}_overall_f1", res["overall_f1"])
            safe_log_metric(f"{split}_overall_mAP50", res["overall_mAP50"])
            safe_log_metric(f"{split}_overall_mAP50_95", res["overall_mAP50_95"])
            safe_log_metric(f"{split}_severity_acc", res["severity_acc"])
            safe_log_metric(f"{split}_severity_mae", res["severity_mae"])

            print(f"    Overall: P={res['overall_precision']:.4f} R={res['overall_recall']:.4f} "
                  f"F1={res['overall_f1']:.4f} mAP50={res['overall_mAP50']:.4f}")
            print(f"    Severity: acc={res['severity_acc']:.4f} mae={res['severity_mae']:.4f}")

            # Per class
            for cname, m in res["per_class"].items():
                safe_log_metric(f"{split}_{cname}_precision", m["precision"])
                safe_log_metric(f"{split}_{cname}_recall", m["recall"])
                safe_log_metric(f"{split}_{cname}_f1", m["f1"])
                safe_log_metric(f"{split}_{cname}_mAP50_95", m["mAP50_95"])
                print(f"      {cname:>10s}: P={m['precision']:.4f} R={m['recall']:.4f} "
                      f"F1={m['f1']:.4f} mAP50-95={m['mAP50_95']:.4f}")

        print(f"\n  Logged MLflow run: {run_name}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def get_existing_run_names():
    """Query MLflow and return run names already finished."""
    try:
        client = mlflow.MlflowClient()
        existing = set()
        for exp in client.search_experiments():
            for run in client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000):
                name = run.data.tags.get("mlflow.runName")
                if name and run.info.status == "FINISHED":
                    existing.add(name)
        print(f"Found {len(existing)} existing FINISHED runs on DagsHub")
        return existing
    except Exception as e:
        print(f"Warning: could not query MLflow: {e}")
        return set()


def main():
    print("Connecting to DagsHub MLflow...")
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    existing = get_existing_run_names()

    # 1. Baselines - val only
    baselines_dir = MODELS_DIR / "baselines"
    if baselines_dir.exists():
        for model_dir in sorted(baselines_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            run_name = f"{model_dir.name.replace('_baseline', '')}_per_class_val"
            if run_name in existing:
                print(f"SKIP (already logged): {run_name}")
                continue
            try:
                log_baseline_val_metrics(model_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    # 2. Multi-task checkpoints - val + test
    for ckpt_dir, run_name in MULTITASK_RUNS:
        if run_name in existing:
            print(f"SKIP (already logged): {run_name}")
            continue
        try:
            log_multitask_val_test_metrics(ckpt_dir, run_name)
        except Exception as e:
            print(f"  ERROR ({ckpt_dir.name}): {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Done. Dashboard: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
