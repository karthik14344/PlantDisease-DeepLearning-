"""
Task 3: Multi-Task YOLOv11 Model (Architecturally Enhanced)
- Standard YOLO detection backbone + neck + head
- CBAM attention on neck features (channel + spatial)
- BiFPN-style weighted fusion for multi-scale features
- Severity-Aware Channel Gating (novel)
- Auxiliary severity classification head (GAP + FC -> 5 classes)
- Combined loss: L_total = L_detection + lambda * L_severity

Architecture name: CBAM-YOLO-MT
"""
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import NUM_CLASSES, NUM_SEVERITY, LAMBDA_SEVERITY
from src.models.attention import CBAM, BiFPNFusion, SeverityAwareGate, SEBlock

logger = logging.getLogger(__name__)


class SeverityHead(nn.Module):
    """
    Auxiliary classification head for severity prediction (0-4).
    Takes multi-scale feature maps (already attended/fused) and classifies.
    """

    def __init__(self, in_channels_list, num_classes=NUM_SEVERITY, dropout=0.3,
                 use_bifpn=True):
        super().__init__()
        self.use_bifpn = use_bifpn

        if use_bifpn:
            # BiFPN fuses multi-scale features into one tensor
            self.bifpn = BiFPNFusion(in_channels_list)
            fused_ch = self.bifpn.out_channels
            self.pool = nn.AdaptiveAvgPool2d(1)
            total_ch = fused_ch
        else:
            # Simple GAP + concat (original approach)
            self.pools = nn.ModuleList(
                [nn.AdaptiveAvgPool2d(1) for _ in in_channels_list]
            )
            total_ch = sum(in_channels_list)

        self.classifier = nn.Sequential(
            nn.Linear(total_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        logger.info(f"SeverityHead: in_channels={in_channels_list}, total={total_ch}, "
                    f"out={num_classes}, use_bifpn={use_bifpn}")

    def forward(self, feature_maps):
        if self.use_bifpn:
            fused = self.bifpn(feature_maps)
            x = self.pool(fused).flatten(1)
        else:
            pooled = [pool(f).flatten(1) for pool, f in zip(self.pools, feature_maps)]
            x = torch.cat(pooled, dim=1)
        return self.classifier(x)


class CBAMNeck(nn.Module):
    """
    Applies CBAM attention to each neck output feature map independently.
    Optionally applies a severity-aware gate using soft severity predictions.
    """

    def __init__(self, in_channels_list, use_severity_gate=True,
                 num_severity=NUM_SEVERITY):
        super().__init__()
        self.cbam_blocks = nn.ModuleList([
            CBAM(ch, reduction=16, kernel_size=7) for ch in in_channels_list
        ])
        self.use_severity_gate = use_severity_gate
        if use_severity_gate:
            self.severity_gates = nn.ModuleList([
                SeverityAwareGate(ch, num_severity=num_severity, embed_dim=32)
                for ch in in_channels_list
            ])
        logger.info(f"CBAMNeck: channels={in_channels_list}, "
                    f"use_severity_gate={use_severity_gate}")

    def forward(self, feature_maps, severity_pred=None):
        """
        Args:
            feature_maps: list of tensors [P3, P4, P5]
            severity_pred: (B, num_severity) soft severity prediction (optional)
        """
        out = []
        for i, feat in enumerate(feature_maps):
            attended = self.cbam_blocks[i](feat)
            if self.use_severity_gate:
                attended = self.severity_gates[i](attended, severity_pred)
            out.append(attended)
        return out


def get_neck_channels(det_model, imgsz=640, device="cpu"):
    """Run a dummy forward pass to auto-detect neck output channel dims."""
    logger.debug("Probing neck channel dimensions...")
    det_model.eval()
    det_model.to(device)
    captured = [None]

    detect_module = det_model.model[-1]
    handle = detect_module.register_forward_pre_hook(
        lambda m, inp: captured.__setitem__(0, inp[0] if isinstance(inp[0], (list, tuple)) else inp)
    )

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        try:
            det_model(dummy)
        except RuntimeError:
            pass

    handle.remove()
    det_model.train()

    if captured[0] is None:
        raise RuntimeError("Failed to capture neck features.")

    channels = [f.shape[1] for f in captured[0]]
    logger.info(f"Neck channels detected: {channels}")
    return channels


class MultiTaskYOLO(nn.Module):
    """
    CBAM-YOLO-MT: Multi-Task YOLO with CBAM attention and BiFPN fusion.

    Architecture:
        Input Image -> [YOLO Backbone] -> [YOLO Neck/FPN] -> [P3, P4, P5]
                                                                  |
                                                          [CBAM + Severity Gate]  (NEW)
                                                                  |
                                                       +----------+-----------+
                                                       |                      |
                                                  [Detect Head]      [BiFPN + Severity Head]  (NEW)
                                                       |                      |
                                                 BBox + Class         Severity 0-4
    """

    def __init__(self, det_model, num_severity=NUM_SEVERITY, dropout=0.3,
                 imgsz=640, device="cpu",
                 use_cbam=True, use_bifpn=True, use_severity_gate=True):
        super().__init__()
        self.det_model = det_model
        self.use_cbam = use_cbam
        self.use_bifpn = use_bifpn
        self.use_severity_gate = use_severity_gate

        # Auto-detect neck channel dimensions
        neck_channels = get_neck_channels(det_model, imgsz=imgsz, device=device)

        # CBAM attention on neck features
        if use_cbam:
            self.cbam_neck = CBAMNeck(
                neck_channels,
                use_severity_gate=use_severity_gate,
                num_severity=num_severity,
            )
        else:
            self.cbam_neck = None

        # Severity head (with optional BiFPN fusion)
        self.severity_head = SeverityHead(
            in_channels_list=neck_channels,
            num_classes=num_severity,
            dropout=dropout,
            use_bifpn=use_bifpn,
        )

        # Pre-hook to capture features flowing into the Detect head
        self._neck_feats = None
        self._hook_handle = self.det_model.model[-1].register_forward_pre_hook(
            self._capture_hook
        )

        # Post-hook to INJECT attended features back before detect head runs
        # We do this by modifying inp in-place via a pre-hook that returns new input
        self._use_attention_injection = use_cbam
        if use_cbam:
            self._inject_handle = self.det_model.model[-1].register_forward_pre_hook(
                self._inject_hook, with_kwargs=False
            )

        logger.info(f"MultiTaskYOLO (CBAM-YOLO-MT) initialized: "
                    f"cbam={use_cbam}, bifpn={use_bifpn}, severity_gate={use_severity_gate}")

    def _capture_hook(self, module, inp):
        """Pre-hook: captures input to Detect head (= neck feature maps)."""
        if isinstance(inp, tuple) and len(inp) > 0:
            feats = inp[0]
            if isinstance(feats, (list, tuple)):
                self._neck_feats = list(feats)
            else:
                self._neck_feats = [feats]

    def _inject_hook(self, module, inp):
        """
        Pre-hook that REPLACES detect head input with CBAM-attended features.
        Returns a modified input tuple.
        """
        # This is called AFTER _capture_hook since they're registered in order
        if self._neck_feats is None or self.cbam_neck is None:
            return None

        # Apply CBAM (without severity gate here — we don't have sev_pred yet)
        # Severity gate is applied in forward() where we have sev_pred
        attended = []
        for i, feat in enumerate(self._neck_feats):
            attended_feat = self.cbam_neck.cbam_blocks[i](feat)
            attended.append(attended_feat)

        # Store attended features for severity head
        self._attended_feats = attended

        # Return modified input (as tuple)
        return (attended,)

    def forward(self, x):
        self._neck_feats = None
        self._attended_feats = None

        # Run YOLO forward (backbone -> neck -> [CBAM injection] -> detect head)
        det_output = self.det_model(x)

        # Use attended features for severity (fall back to raw neck if no CBAM)
        sev_input = self._attended_feats if self._attended_feats is not None else self._neck_feats

        if sev_input is not None:
            # Apply severity-aware gate if enabled
            if self.use_severity_gate and self.cbam_neck is not None:
                # Get initial severity prediction to condition the gate
                sev_logits_init = self.severity_head(sev_input)
                sev_probs = F.softmax(sev_logits_init, dim=1)

                # Re-apply severity gate
                gated = []
                for i, feat in enumerate(sev_input):
                    gated.append(self.cbam_neck.severity_gates[i](feat, sev_probs))
                sev_input = gated

            sev_logits = self.severity_head(sev_input)
        else:
            batch_size = x.shape[0]
            sev_logits = torch.zeros(batch_size, NUM_SEVERITY, device=x.device)

        return det_output, sev_logits

    def remove_hooks(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
        if hasattr(self, '_inject_handle') and self._inject_handle is not None:
            self._inject_handle.remove()


class MultiTaskLoss(nn.Module):
    """Combined loss: L_total = L_detection + lambda * L_severity"""

    def __init__(self, det_loss_fn, lambda_sev=LAMBDA_SEVERITY,
                 severity_loss_type="ce", num_severity=NUM_SEVERITY):
        super().__init__()
        self.det_loss_fn = det_loss_fn
        self.lambda_sev = lambda_sev
        self.severity_loss_type = severity_loss_type

        if severity_loss_type == "focal":
            self.severity_criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif severity_loss_type == "mse":
            self.severity_criterion = nn.MSELoss()
        else:
            self.severity_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        logger.info(f"MultiTaskLoss: lambda={lambda_sev}, severity_loss={severity_loss_type}")

    def forward(self, det_preds, sev_logits, det_batch, severity_labels):
        det_loss, det_loss_items = self.det_loss_fn(det_preds, det_batch)
        if det_loss.dim() > 0:
            det_loss = det_loss.sum()

        valid_mask = severity_labels >= 0
        if valid_mask.any():
            valid_sev_logits = sev_logits[valid_mask]
            valid_sev_labels = severity_labels[valid_mask]
            if self.severity_loss_type == "mse":
                valid_sev_labels = valid_sev_labels.float()
                valid_sev_preds = valid_sev_logits.squeeze(-1)
                sev_loss = self.severity_criterion(valid_sev_preds, valid_sev_labels)
            else:
                sev_loss = self.severity_criterion(valid_sev_logits, valid_sev_labels)
        else:
            sev_loss = torch.tensor(0.0, device=sev_logits.device)

        total_loss = det_loss + self.lambda_sev * sev_loss
        return total_loss, det_loss, sev_loss, det_loss_items


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in severity prediction."""

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_multitask_model(model_name="yolo11n.pt", num_classes=NUM_CLASSES,
                           num_severity=NUM_SEVERITY, device="cpu",
                           use_cbam=True, use_bifpn=True, use_severity_gate=True):
    """Factory function to build the CBAM-YOLO-MT model."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    logger.info(f"Building CBAM-YOLO-MT from {model_name} "
                f"(nc={num_classes}, sev={num_severity})")

    yolo = YOLO(model_name)
    pretrained_sd = yolo.model.state_dict()

    det_model = DetectionModel(cfg=yolo.model.yaml, nc=num_classes, verbose=False)
    model_sd = det_model.state_dict()
    filtered = {k: v for k, v in pretrained_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    det_model.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded {len(filtered)}/{len(model_sd)} pretrained layers")

    model = MultiTaskYOLO(
        det_model=det_model,
        num_severity=num_severity,
        device=device,
        use_cbam=use_cbam,
        use_bifpn=use_bifpn,
        use_severity_gate=use_severity_gate,
    )
    return model
