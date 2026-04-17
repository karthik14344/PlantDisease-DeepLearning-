"""
Attention Modules for Architectural Upgrades
- CBAM (Convolutional Block Attention Module) - Woo et al. 2018
- SE (Squeeze-and-Excitation) - Hu et al. 2018
- BiFPN-style weighted fusion - Tan et al. 2020
- Severity-Aware Channel Gating (novel)
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """Channel attention module from CBAM."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return attention * x


class CBAM(nn.Module):
    """Full CBAM: Channel Attention -> Spatial Attention."""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(in_channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BiFPNFusion(nn.Module):
    """
    BiFPN-style weighted feature fusion for multi-scale features.
    Fuses P3, P4, P5 with learnable weights (normalized via fast softmax).
    """

    def __init__(self, in_channels_list, out_channels=None, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.n_features = len(in_channels_list)

        # Learnable fusion weights (one per scale)
        self.weights = nn.Parameter(torch.ones(self.n_features, dtype=torch.float32))

        # Project all scales to same channel count (use smallest)
        if out_channels is None:
            out_channels = min(in_channels_list)

        self.projs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])
        self.out_channels = out_channels

    def forward(self, feature_maps):
        # Normalize weights via fast softmax (ReLU + epsilon)
        w = F.relu(self.weights)
        w = w / (w.sum() + self.epsilon)

        # Project all features to same channel count
        projected = [proj(f) for proj, f in zip(self.projs, feature_maps)]

        # Resize to largest spatial size (finest scale = P3)
        target_size = projected[0].shape[-2:]
        resized = []
        for f in projected:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            resized.append(f)

        # Weighted sum
        fused = sum(w[i] * resized[i] for i in range(self.n_features))
        return fused


class SeverityAwareGate(nn.Module):
    """
    Severity-Aware Channel Gating (novel).

    Uses a learnable severity embedding to modulate channel attention weights.
    Intuition: different severity levels emphasize different visual features
    (e.g., small lesions vs large necrotic regions).
    """

    def __init__(self, in_channels, num_severity=5, embed_dim=32):
        super().__init__()
        self.severity_embed = nn.Embedding(num_severity + 1, embed_dim)  # +1 for "unknown"
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, severity_pred=None):
        """
        Args:
            x: (B, C, H, W) feature map
            severity_pred: (B, num_severity) soft severity probabilities (optional)
        """
        b, c, _, _ = x.shape

        # If no severity prediction, use mean embedding
        if severity_pred is None:
            sev_idx = torch.zeros(b, dtype=torch.long, device=x.device)
            sev_emb = self.severity_embed(sev_idx)
        else:
            # Soft embedding: weighted sum of severity embeddings
            all_emb = self.severity_embed.weight[:severity_pred.shape[1]]  # (S, D)
            sev_emb = severity_pred @ all_emb  # (B, D)

        # Global feature summary
        x_pool = self.pool(x).view(b, c)

        # Concatenate and produce channel gate
        combined = torch.cat([x_pool, sev_emb], dim=1)
        gate = self.gate_mlp(combined).view(b, c, 1, 1)

        return x * gate
