"""Billiard Balls (Cordonnier et al., "Differentiable Patch Selection", CVPR 2021).

Synthetic 1000x1000 px scenes with 4-7 numbered balls. There are 8000 train, 2000 valid and 10000
test scenes. Each `<id>.png` has a matching `<id>.json`. It lists the balls as
`{"position": [x, y, z], "color", "number"}`.
"""

import json
import pathlib
import tarfile

from PIL import Image
from torch.utils.data import Dataset

from data.augment import eval_transform, train_transform
from data.utils import download

URL = "http://storage.googleapis.com/gresearch/ptokp_patch_selection/billiard.tar.xz"
MD5 = "4bc0b01897385403ab8d1dc8ad37867d"  # 19.7 GB archive and about 37 GB with the extracted images
SPLIT_SIZES = {"train": 8000, "valid": 2000, "test": 10000}


def _leftmost(balls):
    return min(balls, key=lambda b: b["position"][0])


def _rightmost(balls):
    return max(balls, key=lambda b: b["position"][0])


def _left_color_min_max(balls):
    color = _leftmost(balls)["color"]
    if color in {"green", "blue", "purple"}:
        return min(b["number"] for b in balls) - 1
    if color in {"yellow", "red", "orange"}:
        return max(b["number"] for b in balls) - 1
    raise ValueError(f"Unexpected ball color '{color}'.")


# The labels are the ball numbers 1-9 shifted to the classes 0-8.
TASKS = {
    "max-left-right": lambda balls: max(_leftmost(balls)["number"], _rightmost(balls)["number"]) - 1,
    "rightmost-number": lambda balls: _rightmost(balls)["number"] - 1,
    "left-color-min-max": _left_color_min_max,
}
NUM_CLASSES = 9


def _num_images(split_dir):
    return sum(1 for _ in split_dir.glob("*.png")) if split_dir.is_dir() else 0


def prepare(root):
    """Download and extract the dataset into `root`. Nothing happens if all splits are already there."""
    root = pathlib.Path(root)
    if all(_num_images(root / s) == n for s, n in SPLIT_SIZES.items()):
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "billiard.tar.xz"
    download(URL, archive, MD5)
    print(f"Extracting {archive} ...")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(path=root)
    for split, n in SPLIT_SIZES.items():
        if _num_images(root / split) != n:
            raise RuntimeError(f"Expected {n} images in {root / split} after extraction.")


class Billiards(Dataset):
    def __init__(self, root, split, task, img_size, train=False):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}', expected one of {sorted(TASKS)}.")
        self.dir = pathlib.Path(root) / split
        self.label_fn = TASKS[task]
        self.ids = sorted(int(p.stem) for p in self.dir.glob("*.png"))
        # A scale of (1, 1) on square images makes the crop a plain bicubic resize.
        self.transform = train_transform(img_size) if train else eval_transform(img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image = Image.open(self.dir / f"{sample_id}.png").convert("RGB")
        balls = json.loads((self.dir / f"{sample_id}.json").read_text())
        return self.transform(image), self.label_fn(balls)


def load(cfg, split):
    if split not in SPLIT_SIZES:
        raise ValueError(f"Billiard Balls has no '{split}' split, expected one of {list(SPLIT_SIZES)}.")
    return Billiards(cfg.data_root, split, cfg.task, cfg.img_size, train=split == "train")
