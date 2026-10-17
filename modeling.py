"""LookThere with a DINOv2 ViT-B/14 backbone.

The selector is a 3-layer DINOv2. It looks at a low-resolution view of the image and predicts
one logit per high-resolution patch. A hierarchical Bernoulli policy turns these logits into a
keep mask. The extractor is a full 12-layer DINOv2. It processes only the kept high-resolution
patches together with the selector's global tokens (CLS and registers).
"""

import math
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

DINOV2 = "vit_base_patch14_reg4_dinov2.lvd142m"
PATCH_SIZE = 14
SELECTOR_IMG_SIZE = 154  # The selector sees a view of 11x11 patches.
SELECTOR_DEPTH = 3


def _dinov2(img_size, pretrained, **kwargs):
    return timm.create_model(DINOV2, num_classes=0, img_size=img_size, pretrained=pretrained, **kwargs)


def _prefix_tokens(vit, batch_size):
    """The CLS token and the 4 register tokens of DINOv2. They have no positional embedding."""
    return torch.cat([
        vit.cls_token.expand(batch_size, -1, -1),
        vit.reg_token.expand(batch_size, -1, -1),
    ], dim=1)


def mask_to_padded_indices(keep_mask):
    """Convert a (bs, num_patches) bool mask to padded patch indices.

    Returns:
        keep_patch_indices: (bs, k_max) long, padded with 0
        padding_mask:       (bs, k_max) bool, True = real patch
    """
    bs, num_patches = keep_mask.shape
    device = keep_mask.device

    k_counts = keep_mask.sum(dim=-1)
    k_max = k_counts.max().item()

    flat_indices = keep_mask.nonzero(as_tuple=False)

    keep_patch_indices = torch.zeros(bs, k_max, dtype=torch.long, device=device)
    padding_mask = torch.zeros(bs, k_max, dtype=torch.bool, device=device)

    if flat_indices.numel() > 0:
        offsets = torch.zeros(bs, dtype=torch.long, device=device)
        offsets[1:] = k_counts[:-1].cumsum(0)
        pos_in_image = torch.arange(flat_indices.shape[0], device=device) - offsets[flat_indices[:, 0]]

        keep_patch_indices[flat_indices[:, 0], pos_in_image] = flat_indices[:, 1]
        padding_mask[flat_indices[:, 0], pos_in_image] = True

    return keep_patch_indices, padding_mask


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class SelectorHead(nn.Module):
    """MLP that maps each low-res patch token to the logits of the high-res patches it covers."""

    def __init__(self, dim, num_output):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim)
        self.w2 = nn.Linear(4 * dim, num_output)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Selector(nn.Module):
    """Predicts where to look. It gives one keep logit per high-res patch and samples masks from them."""

    def __init__(self, img_size, grid_stages, beta_init=0.25, beta_max=5.0, pretrained=True, logit_offset=0.0):
        super().__init__()
        self.model = _dinov2(SELECTOR_IMG_SIZE, pretrained=False, depth=SELECTOR_DEPTH)
        if pretrained:
            # Use the first 3 blocks of pretrained DINOv2. timm resamples its pos_embed to 11x11.
            full = _dinov2(SELECTOR_IMG_SIZE, pretrained=True).state_dict()
            own = self.model.state_dict()
            self.model.load_state_dict(
                {k: v for k, v in full.items() if k in own and v.shape == own[k].shape}, strict=False,
            )

        self.input_grid = SELECTOR_IMG_SIZE // PATCH_SIZE   # 11
        self.grid = img_size // PATCH_SIZE                  # 72 high-res patches per side at 1008 px
        self.upsample = math.ceil(self.grid / self.input_grid)
        self.head = SelectorHead(self.model.embed_dim, self.upsample ** 2)

        # One learned bias for each step between two levels of the cascade (see `sample`).
        self.betas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(beta_init))) for _ in range(len(grid_stages) - 1)
        ])
        self.beta_max = beta_max
        # Constant added to every keep logit. A positive offset makes the selector keep more patches.
        # It is saved in the checkpoints, so evaluation uses the value the model was trained with.
        self.logit_offset = logit_offset

    def forward(self, images):
        """Return the keep logits (bs, grid*grid) and the prefix tokens (bs, 5, dim)."""
        x = F.interpolate(images, size=(SELECTOR_IMG_SIZE, SELECTOR_IMG_SIZE), mode="bilinear", align_corners=False)
        x = self.model.patch_embed(x) + self.model.pos_embed
        prefix = _prefix_tokens(self.model, x.shape[0])
        num_prefix = prefix.shape[1]
        x = self.model.norm(self.model.blocks(torch.cat([prefix, x], dim=1)))

        logits = rearrange(
            self.head(x[:, num_prefix:]), "b (h w) (i j) -> b 1 (h i) (w j)",
            h=self.input_grid, w=self.input_grid, i=self.upsample, j=self.upsample,
        )
        logits = F.interpolate(logits, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return rearrange(logits, "b 1 h w -> b (h w)") + self.logit_offset, x[:, :num_prefix]

    def sample(self, logits, levels, temperature, deterministic=False):
        """Sample a keep mask with the coarse-to-fine Bernoulli cascade.

        Level k decides on a g_k x g_k grid (for example levels = [6, 18, 37]). It uses
        average-pooled logits. A cell whose parent was dropped at level k-1 gets its logit lowered
        by a learned beta_k. This drops large irrelevant regions together. The finest level
        decides the mask. The mask is upsampled to the patch grid and at least one patch is
        always kept.

        Returns:
            keep_mask: (bs, grid*grid) bool
            log_probs: (bs, sum_k g_k^2) log-probability of every decision
            entropy:   (bs,) summed entropy of all decisions
        """
        bs = logits.shape[0]
        logits_2d = logits.view(bs, 1, self.grid, self.grid)
        log_probs, entropy, actions = [], 0, logits.new_empty(0)

        for k, g in enumerate(levels):
            level_logits = F.adaptive_avg_pool2d(logits_2d, g).view(bs, -1)
            if k > 0:
                beta = self.betas[k - 1].clamp(0.0, self.beta_max)
                parent = F.interpolate(
                    actions.view(bs, 1, levels[k - 1], levels[k - 1]), size=(g, g), mode="nearest"
                ).view(bs, -1)
                level_logits = level_logits - beta * (1.0 - parent)

            dist = torch.distributions.Bernoulli(logits=level_logits / temperature)
            actions: Tensor | Any = (level_logits > 0).float() if deterministic else dist.sample()

            if k == len(levels) - 1:
                empty = actions.sum(dim=-1) == 0
                if empty.any():
                    actions[empty, level_logits[empty].argmax(dim=-1)] = 1.0

            log_probs.append(dist.log_prob(actions))
            entropy = entropy + dist.entropy().sum(dim=-1)

        keep_mask = F.interpolate(
            actions.view(bs, 1, levels[-1], levels[-1]), size=(self.grid, self.grid), mode="nearest"
        ).view(bs, -1).bool()
        return keep_mask, torch.cat(log_probs, dim=-1), entropy


class Extractor(nn.Module):
    """Sees what matters. It is a full DINOv2 that processes only the kept high-res patches.

    The selector computes prefix tokens from the whole low-res image. A learned residual mix
    blends them into the extractor's own prefix tokens. This gives the extractor global context:
        prefix = m * ext_prefix + (1 - m) * LayerScale(W @ sel_prefix)
    The mix m starts at 0.99, W starts as the identity and the LayerScale starts at 1e-4.
    """

    def __init__(self, img_size, num_classes, pretrained=True, drop_path_rate=0.0):
        super().__init__()
        # The registration order sets the parameter order in the optimizer. It is
        # mix_lambda, model, fusion_proj, ls, head. This matches the released optimizer states.
        self.mix_lambda = nn.Parameter(torch.tensor(0.99))
        self.model = _dinov2(img_size, pretrained=pretrained, drop_path_rate=drop_path_rate)
        dim = self.model.embed_dim
        self.fusion_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.fusion_proj.weight)
        self.ls = LayerScale(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, images, selector_prefix, keep_patch_indices, padding_mask):
        """Return the CLS embedding (bs, dim) and the tokens of the kept patches (bs, k_max, dim).

        Apply `self.head` to the CLS embedding to classify.
        """
        x = self.model.patch_embed(images) + self.model.pos_embed
        x = x[torch.arange(x.shape[0], device=x.device)[:, None], keep_patch_indices]
        x = x * padding_mask.unsqueeze(-1).float()

        ext_prefix = _prefix_tokens(self.model, x.shape[0])
        prefix = self.mix_lambda * ext_prefix + (1 - self.mix_lambda) * self.ls(self.fusion_proj(selector_prefix))
        num_prefix = prefix.shape[1]
        x = torch.cat([prefix, x], dim=1)

        # Images keep different numbers of patches. Mask out the padding as attention keys.
        key_mask = torch.cat([
            torch.ones(x.shape[0], num_prefix, device=x.device, dtype=torch.bool),
            padding_mask,
        ], dim=1)
        attn_mask = torch.zeros(x.shape[0], 1, 1, x.shape[1], device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(~key_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        for blk in self.model.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.model.norm(x)
        return {"cls": x[:, 0], "patches": x[:, num_prefix:]}

    def clip_params(self):
        """The parameters under the gradient-norm clip of 1.0. These are all but the classification head."""
        return [p for name, p in self.named_parameters() if not name.startswith("head.")]
