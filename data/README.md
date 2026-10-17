# Data

Set the dataset with `dataset:` in the config. All datasets except ImageNet download to `data_root`
on first use.

| `dataset:` | source | size | train / evaluation split |
|---|---|---|---|
| `billiards` | Cordonnier et al., CVPR 2021 | 19.7 GB (37 GB extracted) | 8,000 train / 2,000 valid / 10,000 test |
| `birdsnap` | Berg et al., CVPR 2014 (HF `sasha/birdsnap`, 500 species) | 69 GB | seeded 90% / 10% |
| `tt100k` | Zhu et al., CVPR 2016 (HF `PrashantDixit0/TT-100K`) | 68 GB | HF `train` + `val` / HF `test` |
| `imagenet` | ILSVRC 2012 | manual | `train` / `val` |

Evaluation uses the `valid` split. `split=test` only exists for Billiard Balls. For the other
datasets `valid` is the single held-out split (the HF `test` split for TT-100K).

## Billiard Balls

Synthetic 1000×1000 px scenes with 4-7 numbered balls. There are three tasks (`task=`):

- `max-left-right`: the larger number of the leftmost and rightmost balls
- `rightmost-number`: the number on the rightmost ball
- `left-color-min-max`: the smallest number if the leftmost ball is green, blue or purple and the largest number otherwise

## Birdsnap

The HuggingFace copy has one split. A seeded 10% of it is held out as `valid`. It needs the
`datasets` package.

## TT-100K

Every sign of at least 32 px becomes a sample. The sample is a 1008 px crop around the sign with the
other signs blacked out. The label is the sign type. Only classes with at least 30 instances are
kept. The class list and the samples are cached in `data_root/index.json`.

## ImageNet

ImageNet is not downloaded. Get it from [image-net.org](https://www.image-net.org/download.php),
arrange it as below and set `data_root`:

```
imagenet/
  train/<wnid>/*.JPEG      # 1000 class folders
  val/<wnid>/*.JPEG        # `validation/` also works
```

The standard validation archive is flat. The script from the
[PyTorch ImageNet example](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)
sorts it into class folders.

## Augmentation

Training uses a random-resized crop and RandAugment(2, 5). TT-100K has no crop and ImageNet uses
3-Augment. `augment:` selects a preset in `data/<dataset>.py`. Setting `cutmix_alpha` and
`label_smoothing` (for example `1.0` and `0.1` with `augment=heavy` on Birdsnap) adds CutMix and
label smoothing. Evaluation resizes the shorter side to `img_size` and takes a center crop.
