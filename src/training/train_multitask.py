"""
Task 4b: Multi-Task Training
- Custom PyTorch training loop for YOLO + Severity joint training
- Combined loss: L_detection + lambda * L_severity
- Logging, early stopping, checkpoint saving
"""
import sys
import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PROCESSED_DIR, MODELS_DIR, NUM_CLASSES, NUM_SEVERITY,
    IMG_SIZE, BATCH_SIZE, EPOCHS_MULTITASK, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_EPOCHS, PATIENCE, LAMBDA_SEVERITY,
    DEVICE, NUM_WORKERS, SEVERITY_NOT_ESTIMABLE,
    USE_CBAM, USE_BIFPN, USE_SEVERITY_GATE,
    INIT_MULTITASK_FROM_BASELINE, BASELINE_CHECKPOINT,
)
from src.models.multitask_yolo import (
    MultiTaskYOLO, MultiTaskLoss, SeverityHead, get_neck_channels,
)

logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """PyTorch Dataset that loads images with both YOLO targets and severity labels."""

    def __init__(self, images_dir, labels_dir, severity_map, imgsz=IMG_SIZE,
                 augment=False):
        self.imgsz = imgsz
        self.augment = augment
        self.severity_map = severity_map

        self.samples = []
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_path.stem
            lbl_path = labels_dir / f"{stem}.txt"
            if lbl_path.exists():
                sev_info = severity_map.get(stem, {})
                severity = sev_info.get("severity", SEVERITY_NOT_ESTIMABLE)
                self.samples.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "stem": stem,
                    "severity": severity,
                })

        logger.info(f"MultiTaskDataset: {len(self.samples)} samples from {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = letterbox(img, self.imgsz)

        labels = []
        with open(sample["label_path"], "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    labels.append([cls_id, x_c, y_c, w, h])

        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        if self.augment:
            img, labels = basic_augment(img, labels)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        labels_tensor = torch.from_numpy(labels)
        severity = sample["severity"]

        return img_tensor, labels_tensor, severity, idx

    @staticmethod
    def collate_fn(batch):
        imgs, labels_list, severities, indices = zip(*batch)
        imgs = torch.stack(imgs, 0)
        severities = torch.tensor(severities, dtype=torch.long)

        batch_labels = []
        for i, labels in enumerate(labels_list):
            if len(labels) > 0:
                batch_idx = torch.full((len(labels), 1), i, dtype=torch.float32)
                batch_labels.append(torch.cat([batch_idx, labels], dim=1))

        if batch_labels:
            batch_labels = torch.cat(batch_labels, 0)
        else:
            batch_labels = torch.zeros((0, 6), dtype=torch.float32)

        return imgs, batch_labels, severities, indices


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def basic_augment(img, labels):
    if np.random.random() < 0.5:
        img = np.fliplr(img).copy()
        if len(labels) > 0:
            labels[:, 1] = 1.0 - labels[:, 1]
    if np.random.random() < 0.5:
        h_gain, s_gain, v_gain = 0.015, 0.7, 0.4
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        x = np.arange(0, 256, dtype=np.float32)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        img_hsv = cv2.merge([cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)])
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img, labels


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"EarlyStopping triggered after {self.counter} epochs without improvement")
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def train_one_epoch(model, det_loss_fn, dataloader, optimizer, device,
                    lambda_sev=LAMBDA_SEVERITY, epoch=0, warmup_epochs=3):
    model.train()
    metrics = defaultdict(float)
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for imgs, batch_labels, severities, _ in pbar:
        imgs = imgs.to(device)
        batch_labels = batch_labels.to(device)
        severities = severities.to(device)

        warmup_lr(optimizer, epoch, warmup_epochs, LEARNING_RATE)

        det_preds, sev_logits = model(imgs)

        det_batch = {
            "batch_idx": batch_labels[:, 0].long() if len(batch_labels) > 0 else torch.zeros(0, dtype=torch.long, device=device),
            "cls": batch_labels[:, 1:2] if len(batch_labels) > 0 else torch.zeros((0, 1), device=device),
            "bboxes": batch_labels[:, 2:6] if len(batch_labels) > 0 else torch.zeros((0, 4), device=device),
        }
        det_loss, det_loss_items = det_loss_fn(det_preds, det_batch)
        if det_loss.dim() > 0:
            det_loss = det_loss.sum()

        valid_mask = severities >= 0
        if valid_mask.any():
            sev_loss = F.cross_entropy(sev_logits[valid_mask], severities[valid_mask], label_smoothing=0.05)
        else:
            sev_loss = torch.tensor(0.0, device=device)

        total_loss = det_loss + lambda_sev * sev_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        metrics["total_loss"] += total_loss.item()
        metrics["det_loss"] += det_loss.item()
        metrics["sev_loss"] += sev_loss.item()

        if valid_mask.any():
            sev_preds = sev_logits[valid_mask].argmax(dim=1)
            sev_correct = (sev_preds == severities[valid_mask]).float().mean().item()
            metrics["sev_acc"] += sev_correct
            metrics["sev_count"] += 1

        n_batches += 1
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "det": f"{det_loss.item():.4f}", "sev": f"{sev_loss.item():.4f}"})

    for k in metrics:
        if k != "sev_count":
            metrics[k] /= max(n_batches, 1)
    if metrics["sev_count"] > 0:
        metrics["sev_acc"] /= metrics["sev_count"]

    return dict(metrics)


@torch.no_grad()
def validate(model, det_loss_fn, dataloader, device, lambda_sev=LAMBDA_SEVERITY):
    model.eval()
    metrics = defaultdict(float)
    n_batches = 0
    all_sev_preds, all_sev_labels = [], []

    for imgs, batch_labels, severities, _ in dataloader:
        imgs = imgs.to(device)
        batch_labels = batch_labels.to(device)
        severities = severities.to(device)

        det_preds, sev_logits = model(imgs)

        det_batch = {
            "batch_idx": batch_labels[:, 0].long() if len(batch_labels) > 0 else torch.zeros(0, dtype=torch.long, device=device),
            "cls": batch_labels[:, 1:2] if len(batch_labels) > 0 else torch.zeros((0, 1), device=device),
            "bboxes": batch_labels[:, 2:6] if len(batch_labels) > 0 else torch.zeros((0, 4), device=device),
        }
        det_loss, _ = det_loss_fn(det_preds, det_batch)
        if det_loss.dim() > 0:
            det_loss = det_loss.sum()

        valid_mask = severities >= 0
        if valid_mask.any():
            sev_loss = F.cross_entropy(sev_logits[valid_mask], severities[valid_mask])
            sev_preds = sev_logits[valid_mask].argmax(dim=1)
            all_sev_preds.extend(sev_preds.cpu().tolist())
            all_sev_labels.extend(severities[valid_mask].cpu().tolist())
        else:
            sev_loss = torch.tensor(0.0, device=device)

        metrics["det_loss"] += det_loss.item()
        metrics["sev_loss"] += sev_loss.item()
        metrics["total_loss"] += (det_loss + lambda_sev * sev_loss).item()
        n_batches += 1

    for k in metrics:
        metrics[k] /= max(n_batches, 1)

    if all_sev_preds:
        all_sev_preds = np.array(all_sev_preds)
        all_sev_labels = np.array(all_sev_labels)
        metrics["sev_acc"] = float((all_sev_preds == all_sev_labels).mean())
        metrics["sev_mae"] = float(np.abs(all_sev_preds - all_sev_labels).mean())
    else:
        metrics["sev_acc"] = 0.0
        metrics["sev_mae"] = 0.0

    return dict(metrics)


def run_multitask_training(model_name="yolo11n.pt", lambda_sev=LAMBDA_SEVERITY,
                            run_name=None, resume_dir=None):
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    logger.info("#" * 60)
    logger.info(f"TASK 4b: MULTI-TASK TRAINING (lambda={lambda_sev})")
    logger.info("#" * 60)

    device = DEVICE
    data_dir = PROCESSED_DIR
    severity_path = data_dir / "severity_labels.json"

    with open(severity_path, "r") as f:
        severity_map = json.load(f)
    logger.info(f"Loaded {len(severity_map)} severity labels")

    train_ds = MultiTaskDataset(
        images_dir=data_dir / "images" / "train",
        labels_dir=data_dir / "labels" / "train",
        severity_map=severity_map, imgsz=IMG_SIZE, augment=True,
    )
    val_ds = MultiTaskDataset(
        images_dir=data_dir / "images" / "val",
        labels_dir=data_dir / "labels" / "val",
        severity_map=severity_map, imgsz=IMG_SIZE, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=MultiTaskDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=MultiTaskDataset.collate_fn)

    from ultralytics.nn.tasks import DetectionModel

    if INIT_MULTITASK_FROM_BASELINE and BASELINE_CHECKPOINT.exists():
        # TWO-STAGE STRATEGY: load DiaMOS-trained YOLOv11n baseline (already nc=4)
        logger.info(f"Two-stage init: loading DiaMOS baseline from {BASELINE_CHECKPOINT}")
        yolo_baseline = YOLO(str(BASELINE_CHECKPOINT))
        pretrained_sd = yolo_baseline.model.state_dict()

        # Build fresh DetectionModel with nc=4 (matching architecture)
        det_model = DetectionModel(cfg=yolo_baseline.model.yaml, nc=NUM_CLASSES, verbose=False)
        model_sd = det_model.state_dict()
        filtered = {k: v for k, v in pretrained_sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        det_model.load_state_dict(filtered, strict=False)
        logger.info(f"  Loaded {len(filtered)}/{len(model_sd)} layers from DiaMOS baseline "
                    f"(nc={NUM_CLASSES}). Detection layers already pear-aware.")
    else:
        # END-TO-END STRATEGY: load generic COCO pretrained weights
        logger.info(f"End-to-end init: loading COCO pretrained from {model_name}")
        yolo = YOLO(model_name)
        pretrained_sd = yolo.model.state_dict()
        det_model = DetectionModel(cfg=yolo.model.yaml, nc=NUM_CLASSES, verbose=False)
        model_sd = det_model.state_dict()
        filtered = {k: v for k, v in pretrained_sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        det_model.load_state_dict(filtered, strict=False)
        logger.info(f"  Loaded {len(filtered)}/{len(model_sd)} layers from COCO (nc={NUM_CLASSES})")

    # Attach required args for v8DetectionLoss
    from types import SimpleNamespace
    det_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    logger.info(f"Architecture flags: CBAM={USE_CBAM}, BiFPN={USE_BIFPN}, SevGate={USE_SEVERITY_GATE}")
    model = MultiTaskYOLO(
        det_model=det_model, num_severity=NUM_SEVERITY, device=str(device),
        use_cbam=USE_CBAM, use_bifpn=USE_BIFPN, use_severity_gate=USE_SEVERITY_GATE,
    )
    model = model.to(device)

    det_loss_fn = v8DetectionLoss(det_model)

    backbone_params = list(model.det_model.parameters())
    severity_params = list(model.severity_head.parameters())
    # Attention modules (CBAM + severity gates) - higher LR since trained from scratch
    attention_params = []
    if model.cbam_neck is not None:
        attention_params += list(model.cbam_neck.parameters())

    param_groups = [
        {"params": backbone_params, "lr": LEARNING_RATE},
        {"params": severity_params, "lr": LEARNING_RATE * 5},
    ]
    if attention_params:
        param_groups.append({"params": attention_params, "lr": LEARNING_RATE * 5})

    optimizer = torch.optim.SGD(
        param_groups, momentum=0.937, weight_decay=WEIGHT_DECAY, nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_MULTITASK, eta_min=LEARNING_RATE * 0.01
    )
    early_stop = EarlyStopping(patience=PATIENCE)

    if resume_dir is not None:
        output_dir = Path(resume_dir)
    else:
        if run_name is None:
            init_tag = "twostage" if INIT_MULTITASK_FROM_BASELINE and BASELINE_CHECKPOINT.exists() else "cocoinit"
            run_name = f"yolo11n_multitask_{init_tag}_lambda{lambda_sev}_{datetime.now():%Y%m%d_%H%M}"
        output_dir = MODELS_DIR / "multitask" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if last.pt exists
    start_epoch = 0
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_epoch = 0

    resume_path = output_dir / "last.pt"
    if resume_path.exists():
        logger.info(f"RESUMING from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Advance scheduler to match
        for _ in range(start_epoch):
            scheduler.step()
        # Load existing history
        hist_path = output_dir / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            best_val = min(history["val"], key=lambda x: x["total_loss"])
            best_val_loss = best_val["total_loss"]
            best_epoch = history["val"].index(best_val) + 1
        logger.info(f"Resumed at epoch {start_epoch + 1}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        logger.info(f"Starting fresh training: epochs={EPOCHS_MULTITASK}, output={output_dir}, device={device}")

    for epoch in range(start_epoch, EPOCHS_MULTITASK):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, det_loss_fn, train_loader, optimizer, device,
            lambda_sev=lambda_sev, epoch=epoch, warmup_epochs=WARMUP_EPOCHS,
        )
        val_metrics = validate(model, det_loss_fn, val_loader, device, lambda_sev=lambda_sev)

        scheduler.step()
        elapsed = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        logger.info(
            f"Epoch {epoch+1:>3d}/{EPOCHS_MULTITASK} | "
            f"Train: {train_metrics['total_loss']:.4f} (det={train_metrics['det_loss']:.4f} sev={train_metrics['sev_loss']:.4f}) | "
            f"Val: {val_metrics['total_loss']:.4f} | "
            f"Sev Acc: {val_metrics['sev_acc']:.3f} MAE: {val_metrics['sev_mae']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics, "lambda_sev": lambda_sev,
            }, output_dir / "best.pt")
            logger.debug(f"New best model saved at epoch {best_epoch}")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics, "lambda_sev": lambda_sev,
        }, output_dir / "last.pt")

        # Save history after every epoch (so resume always has it)
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if early_stop(-val_metrics["total_loss"]):
            logger.info(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    logger.info(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved: {output_dir / 'best.pt'}")
    logger.info("MULTI-TASK TRAINING COMPLETE")

    return output_dir, history


if __name__ == "__main__":
    run_multitask_training()
