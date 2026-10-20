# LookThere! Sparse Vision by Reinforced Selection

Official code for **LookThere! Sparse Vision by Reinforced Selection** (NeurIPS 2026)<br>
Sreehari Rammohan\*, Yousef Yassin\*, Anthony Fuller, Junfeng Wen, Carl Vondrick, Evan Shelhamer

LookThere splits a vision transformer into a shallow **selector** and a deep **extractor**. The
selector looks at a low-resolution view of the image and samples which high-resolution patches to
keep. The extractor processes only those patches. The extractor learns with supervised learning and
the selector with reinforcement learning (GRPO). The selector is rewarded for sparse masks that keep
the loss of the extractor low.

This repository has the DINOv2 ViT-B/14 model and the training code. On Billiard Balls (1008×1008 px
images with 72×72 patches) LookThere reaches 97% accuracy with 5% of the patches.

| file | contents |
|---|---|
| `modeling.py` | `Selector` (3-layer DINOv2 on a 154 px view, hierarchical Bernoulli policy) and `Extractor` (DINOv2 ViT-B/14 on the kept patches) |
| `objectives.py` | classification and DINOv2 distillation losses |
| `train.py` | warmup, RL fine-tuning (GRPO) and evaluation, on one GPU or several with `torchrun` |
| `data/` | datasets, augmentations and CutMix (see [data/README.md](data/README.md)) |
| `configs/<dataset>/` | warmup and RL settings of selected paper runs |

## Setup

```bash
pip install -r requirements.txt   # Python 3.11
```

Training uses bf16. RL at batch size 32 needs about 50 GB of GPU memory (we used 80 GB A100s).
The commands below use one GPU. To use several, replace `python` with `torchrun --nproc_per_node=N`.
`batch_size` is per GPU, so the total batch size is N × `batch_size`.

## Usage

Run the examples from the repository root. The checkpoints of the paper are at [huggingface.co/SreehariRamMohan/lookthere](https://huggingface.co/SreehariRamMohan/lookthere).

```python
import torch
from PIL import Image
from torchvision import transforms
from modeling import Extractor, Selector, mask_to_padded_indices

# user settings
checkpoint_path = "lookthere_ckpts/billiards/max-left-right/sparsity90/model.pt"  # hf download SreehariRamMohan/lookthere --local-dir lookthere_ckpts
image_path = "image.jpg"
deterministic = False  # True keeps exactly the patches with a probability above 0.5
device = "cuda"

# load the checkpoint. It stores the settings it was trained with
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
cfg = ckpt["config"]
num_classes = ckpt["extractor"]["head.weight"].shape[0]

selector = Selector(cfg["img_size"], cfg["grid_stages"], pretrained=False).to(device).eval()
extractor = Extractor(cfg["img_size"], num_classes, pretrained=False).to(device).eval()
selector.load_state_dict(ckpt["selector"])
extractor.load_state_dict(ckpt["extractor"])
selector.logit_offset = ckpt.get("logit_offset", 0.0)  # the offset the model was trained with (3.0 for ImageNet 90%)

# prepare input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(cfg["img_size"], antialias=True),
    transforms.CenterCrop(cfg["img_size"]),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)  # (1, 3, img_size, img_size)

# select the patches and classify
with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
    logits, prefix = selector(image)  # (1, num_patches)
    keep_mask, _, _ = selector.sample(logits, cfg["grid_stages"], ckpt["temperature"], deterministic)
    keep_idx, pad = mask_to_padded_indices(keep_mask)
    out = extractor(image, prefix, keep_idx, pad)
    prediction = extractor.head(out["cls"]).argmax(dim=-1)  # (1,)
    # out["cls"]      (1, 768)     embedding of the image
    # out["patches"]  (1, k, 768)  tokens of the kept patches
```

## Data

Set the dataset with `dataset:` in the config. The datasets are Billiard Balls, Birdsnap, TT-100K
and ImageNet. All except ImageNet download to `data_root` on first use. See
[data/README.md](data/README.md) for sizes, splits, preprocessing and augmentation.

## Training

Training has two phases. Any config key can be overridden as `key=value`. The commands are for
Billiard Balls. The other datasets work the same way, and each RL config names its starting
checkpoint in `init:`.

**1. Warmup.** The extractor learns the task from all patches (25k steps at batch size 16):

```bash
python train.py warmup configs/billiards/billiards_warmup.yaml task=max-left-right output_dir=runs/warmup_max-left-right
```

**2. RL.** The selector learns where to look with GRPO and the extractor keeps learning from the
patches it gets (10k steps at batch size 32, about 3 hours on an A100):

```bash
python train.py rl configs/billiards/billiards_95_rl.yaml task=max-left-right \
    init=runs/warmup_max-left-right/warmup.pt output_dir=runs/rl_billiards_95
```

The paper's warmup checkpoints are at [huggingface.co/SreehariRamMohan/lookthere](https://huggingface.co/SreehariRamMohan/lookthere). Pass one as `init=` to skip the warmup.

The other warmups follow the paper's runs: TT-100K uses 8430 steps at batch size 32, Birdsnap 30
epochs at batch size 32 and ImageNet 125k steps at batch size 64 on 8 GPUs:

```bash
torchrun --nproc_per_node=8 train.py warmup configs/imagenet/imagenet_warmup.yaml
```

The paper's Birdsnap 95% run started from a checkpoint distilled on ImageNet
(see Distillation).

For each image the selector samples 8 masks. The reward of a mask is the negative loss of the
extractor plus `lam` times a sparsity reward. For classification the sparsity reward is only given
if the prediction is correct. It peaks at the keep rate `sparsity_target` (see
`train.sparsity_reward`). The policy is a coarse-to-fine cascade of Bernoulli decisions on 6×6,
18×18 and 37×37 grids. They are added one at a time (`grid_stages`, `steps_per_stage`).

## Distillation

Set `loss: distillation` to train without labels. The selector and the extractor start from
pretrained DINOv2. The extractor matches the CLS and patch tokens of a frozen DINOv2 teacher that
sees the full image. Dropped patches are filled in from the nearest kept ones before the
comparison. The reward is the negative distillation loss plus the sparsity reward. Its keep rate
target ramps linearly from 0.5 to 0.3.

```bash
torchrun --nproc_per_node=8 train.py rl configs/imagenet/imagenet_distill_dinov2.yaml
```

It needs no warmup and no `init`. Pass its final checkpoint as `init` of another RL config to load
the selector and extractor weights. For classification, the new head is then initialized with a
linear probe (logistic regression on the CLS tokens with all patches kept), as in the paper.
`linear_probe=false` starts it from random weights instead.

## Evaluation

```bash
python train.py eval configs/billiards/billiards_95_rl.yaml ckpt=runs/rl_billiards_95/final.pt
```

Masks are sampled at the final temperature of training. `deterministic=true` keeps the patches with
a probability above 0.5. `split=test` is only available for Billiard Balls. The other datasets have
one held-out split, `valid` (the HF `test` split for TT-100K). Only the evaluated split is loaded.

RL training evaluates on `valid` after every epoch. It writes `summary.json` with the final and the
best accuracy (the loss for distillation) and the matching sparsity.

## Reproducing the paper

The paper reports single runs on the `valid` split: the best accuracy over epochs (`best_val_acc` in
`summary.json`) and the sparsity of that evaluation. Each Billiards point is the best run of a sweep
over the keep rate target `k` and `lam` ∈ {2, 5}. The table is for max-left-right. Dense DINOv2 on
all patches reaches 97.7%.

| sparsity | overrides | accuracy |
|---|---|---|
| 50% | `sparsity_target=0.5 lam=2` | 97.7 |
| 75% | `sparsity_target=0.25 lam=5` | 97.6 |
| 95% | defaults (`sparsity_target=0.05 lam=5`) | 97.0 |
| 99% | `sparsity_target=0.01 lam=5` | 94.3 |
| 99.4% | `sparsity_target=0.001 lam=2` | 90.3 |

## Citation

```bibtex
@inproceedings{rammohan2026lookthere,
  title     = {LookThere! Sparse Vision by Reinforced Selection},
  author    = {Rammohan, Sreehari and Yassin, Yousef and Fuller, Anthony and Wen, Junfeng and Vondrick, Carl and Shelhamer, Evan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

LookThere builds on [LookWhere](https://github.com/antofuller/lookwhere) (Fuller et al., NeurIPS
2025). The datasets are:

- Billiard Balls: [Differentiable Patch Selection](https://github.com/google-research/google-research/tree/master/ptopk_patch_selection)
  (Cordonnier et al., CVPR 2021)
- Birdsnap: [Birdsnap: Large-scale Fine-grained Visual Categorization of Birds](https://huggingface.co/datasets/sasha/birdsnap)
  (Berg et al., CVPR 2014)
- TT-100K: [Traffic-Sign Detection and Classification in the Wild](https://huggingface.co/datasets/PrashantDixit0/TT-100K)
  (Zhu et al., CVPR 2016)
- ImageNet: [ImageNet Large Scale Visual Recognition Challenge](https://www.image-net.org/)
  (Russakovsky et al., IJCV 2015)
