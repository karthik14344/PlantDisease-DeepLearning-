"""
Quick evaluation of the two-stage CBAM-YOLO-MT checkpoint on val + test.
Prints per-class P/R/F1/mAP and severity acc/MAE to stdout (no MLflow).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scripts.log_per_class_metrics import evaluate_multitask_on_split
from src.config import (
    MODELS_DIR, NUM_CLASSES, NUM_SEVERITY, DEVICE, CLASS_NAMES,
    USE_CBAM, USE_BIFPN, USE_SEVERITY_GATE,
)

CKPT_DIR = MODELS_DIR / "multitask" / "yolo11n_multitask_twostage_lambda0.5_20260502_1029"


def main():
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace
    from src.models.multitask_yolo import MultiTaskYOLO

    best_pt = CKPT_DIR / "best.pt"
    print(f"Loading checkpoint: {best_pt}")
    print(f"Device: {DEVICE}")

    yolo = YOLO("yolo11n.pt")
    det_model = DetectionModel(cfg=yolo.model.yaml, nc=NUM_CLASSES, verbose=False)
    det_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    model = MultiTaskYOLO(
        det_model=det_model, num_severity=NUM_SEVERITY, device=str(DEVICE),
        use_cbam=USE_CBAM, use_bifpn=USE_BIFPN, use_severity_gate=USE_SEVERITY_GATE,
    ).to(DEVICE)

    ckpt = torch.load(best_pt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded epoch: {ckpt.get('epoch')}")
    det_loss_fn = v8DetectionLoss(det_model)

    for split in ["val", "test"]:
        print(f"\n{'=' * 60}\nSplit: {split}\n{'=' * 60}")
        res = evaluate_multitask_on_split(model, det_loss_fn, split, DEVICE)
        print(f"\n  Overall: P={res['overall_precision']:.4f} "
              f"R={res['overall_recall']:.4f} F1={res['overall_f1']:.4f} "
              f"mAP@0.5={res['overall_mAP50']:.4f} mAP@0.5:0.95={res['overall_mAP50_95']:.4f}")
        print(f"  Severity: acc={res['severity_acc']:.4f} mae={res['severity_mae']:.4f}")
        print(f"\n  Per class:")
        print(f"    {'class':>10s}  {'P':>7s} {'R':>7s} {'F1':>7s} {'mAP50-95':>9s}  {'TP':>4s} {'FP':>4s} {'FN':>4s}")
        for cname, m in res["per_class"].items():
            print(f"    {cname:>10s}  {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f}  "
                  f"{m['mAP50_95']:.4f}    {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d}")


if __name__ == "__main__":
    main()
