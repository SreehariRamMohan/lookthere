"""The training objectives. Classification trains on labels and distillation trains on a DINOv2 teacher.

An objective turns the output of the extractor into a loss for each image. The RL reward of a mask is
built from this loss. Every method works on batches and runs under the autocast of the training step.
"""

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification:
    """Cross-entropy on the CLS token. The sparsity reward is only given for correct predictions."""

    metric, short_name, sign = "accuracy", "acc", 1.0  # The best checkpoint has the highest accuracy.

    def targets(self, images, labels):
        return labels

    def loss(self, extractor, out, labels, keep_idx, pad):
        return F.cross_entropy(extractor.head(out["cls"]), labels, reduction="none")

    def gate(self, extractor, out, labels):
        """1 for the images the extractor classifies correctly. CutMix labels count by their argmax."""
        hard = labels.argmax(dim=-1) if labels.ndim == 2 else labels
        return (extractor.head(out["cls"]).argmax(dim=-1) == hard).float()

    def eval_metrics(self, extractor, out, labels, keep_idx, pad):
        return {"loss": self.loss(extractor, out, labels, keep_idx, pad),
                "accuracy": self.gate(extractor, out, labels)}


class Teacher(nn.Module):
    """Frozen DINOv2 ViT-B/14. It returns the CLS token and the patch tokens of the full image as a dict."""

    def __init__(self, img_size, pretrained=True):
        super().__init__()
        self.model: Any = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=pretrained,
                                            num_classes=0, img_size=img_size)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        x = self.model.forward_features(images)
        return {"cls": x[:, 0], "patches": x[:, self.model.num_prefix_tokens:]}


def upsample_patches(keep_idx, values, pad, grid, k=5):
    """Fill the dropped patches of a grid x grid map with an inverse-distance average of the k nearest kept ones.

    keep_idx: (bs, n) indices of the kept patches. values: (bs, n, dim). pad: (bs, n) bool, True = real patch.
    Returns (bs, grid * grid, dim).
    """
    bs, n = keep_idx.shape
    rows, cols = torch.meshgrid(torch.arange(grid, device=keep_idx.device),
                                torch.arange(grid, device=keep_idx.device), indexing="ij")
    cells = torch.stack([rows, cols], dim=-1).view(1, -1, 1, 2)                          # (1, grid^2, 1, 2)
    kept = torch.stack([keep_idx // grid, keep_idx % grid], dim=-1).unsqueeze(1)         # (bs, 1, n, 2)
    dists = (cells - kept).float().pow(2).sum(-1).sqrt()                                 # (bs, grid^2, n)
    dists = dists.masked_fill(~pad.unsqueeze(1), float("inf"))                           # Never pick padding.
    nearest = dists.argsort(dim=-1, stable=True)[:, :, :k]                               # (bs, grid^2, k)
    weights = 1.0 / (torch.gather(dists, -1, nearest) + 1e-8)
    weights = weights / weights.sum(-1, keepdim=True)
    batch = torch.arange(bs, device=keep_idx.device).view(bs, 1, 1)
    return (values[batch, nearest] * weights.unsqueeze(-1)).sum(dim=2)


class Distillation:
    """Match the CLS token and the patch tokens of a frozen DINOv2 teacher that sees the full image.

    The loss is cls_weight * MSE(CLS) + patch_weight * MSE(patches). The patches of the student are
    upsampled to the full grid first, so that keeping fewer patches does not make the loss smaller.
    The sparsity reward is given for every image.
    """

    metric, short_name, sign = "loss", "loss", -1.0  # The best checkpoint has the lowest loss.

    def __init__(self, teacher, cls_weight, patch_weight):
        self.teacher, self.cls_weight, self.patch_weight = teacher, cls_weight, patch_weight

    def targets(self, images, labels):
        return self.teacher(images)

    def components(self, extractor, out, targets, keep_idx, pad):
        grid = int(targets["patches"].shape[1] ** 0.5)
        full = upsample_patches(keep_idx, out["patches"], pad, grid)
        cls_loss = F.mse_loss(out["cls"], targets["cls"], reduction="none").mean(-1)
        patch_loss = F.mse_loss(full, targets["patches"], reduction="none").mean(-1).mean(-1)
        return cls_loss, patch_loss

    def loss(self, extractor, out, targets, keep_idx, pad):
        cls_loss, patch_loss = self.components(extractor, out, targets, keep_idx, pad)
        return self.cls_weight * cls_loss + self.patch_weight * patch_loss

    def gate(self, extractor, out, targets):
        # No gating is applied for distillation; always return 1.0.
        return torch.ones(out["cls"].shape[0], device=out["cls"].device)

    def eval_metrics(self, extractor, out, targets, keep_idx, pad):
        cls_loss, patch_loss = self.components(extractor, out, targets, keep_idx, pad)
        return {"loss": self.cls_weight * cls_loss + self.patch_weight * patch_loss,
                "cls_loss": cls_loss, "patch_loss": patch_loss}


def make_objective(cfg):
    if cfg.loss == "classification":
        return Classification()
    if cfg.loss == "distillation":
        return Distillation(Teacher(cfg.img_size).cuda(), cfg.cls_weight, cfg.patch_weight)
    raise ValueError(f"Unknown loss '{cfg.loss}', expected 'classification' or 'distillation'.")
