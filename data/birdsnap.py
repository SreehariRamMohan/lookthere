"""Birdsnap (Berg et al., CVPR 2014): fine-grained recognition of 500 North American bird species.

We use the HuggingFace copy `sasha/birdsnap` (about 69 GB). It has one train split with an `image`
and a `label` column. There is no official split, so we hold out a seeded 10% as `valid`.
"""

from PIL import ImageFile
from torch.utils.data import Dataset

from data.augment import eval_transform, train_transform

setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)  # A few files in the dataset are truncated.

HF_REPO = "sasha/birdsnap"
NUM_CLASSES = 500
VAL_FRACTION, SPLIT_SEED = 0.1, 42
# The "heavy" preset is meant to be used with cutmix_alpha=1.0 and label_smoothing=0.1.
AUGMENT = {
    "default": dict(scale=(0.5, 1.0), flip=True),
    "heavy": dict(scale=(0.5, 1.0), flip=True, erase=0.1),
}


def prepare(root):
    """Download the dataset into `root`. Nothing happens if it is already cached."""
    from datasets import load_dataset
    load_dataset(HF_REPO, split="train", cache_dir=str(root))


class Birdsnap(Dataset):
    def __init__(self, images, vocab, transform):
        self.images, self.vocab, self.transform = images, vocab, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = self.images[index]
        return self.transform(item["image"].convert("RGB")), self.vocab[item["label"]]


def load(cfg, split):
    from datasets import load_dataset
    full = load_dataset(HF_REPO, split="train", cache_dir=str(cfg.data_root))
    vocab = {name: i for i, name in enumerate(sorted(set(full["label"])))}
    full = full.shuffle(seed=SPLIT_SEED)
    num_train = len(full) - int(len(full) * VAL_FRACTION)
    if split == "train":
        return Birdsnap(full.select(range(num_train)), vocab, train_transform(cfg.img_size, **AUGMENT[cfg.augment]))
    if split == "valid":
        return Birdsnap(full.select(range(num_train, len(full))), vocab, eval_transform(cfg.img_size))
    raise ValueError(f"Birdsnap has no '{split}' split, expected 'train' or 'valid'.")
