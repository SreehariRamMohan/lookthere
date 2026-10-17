"""ImageNet-1k (ILSVRC 2012).

It is not downloaded. Point `data_root` at an existing copy with the layout
`train/<wnid>/*.JPEG` and `val/<wnid>/*.JPEG`. A `validation/` folder also works. See the README.
"""

import pathlib

from torchvision.datasets import ImageFolder

from data.augment import eval_transform, three_augment

NUM_CLASSES = 1000


def _split_dir(root, split):
    if split not in ("train", "valid"):
        raise ValueError(f"ImageNet has no '{split}' split, expected 'train' or 'valid'.")
    names = ["train"] if split == "train" else ["val", "validation"]
    for name in names:
        if (pathlib.Path(root) / name).is_dir():
            return pathlib.Path(root) / name
    raise FileNotFoundError(
        f"No {'/'.join(names)} directory in {root}. ImageNet is not downloaded automatically: "
        "set data_root to an existing copy (see the README)."
    )


def prepare(root):
    """Nothing is downloaded. `load` reports a missing folder."""


def load(cfg, split):
    transform = three_augment(cfg.img_size) if split == "train" else eval_transform(cfg.img_size)
    return ImageFolder(str(_split_dir(cfg.data_root, split)), transform=transform)
