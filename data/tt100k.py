"""TT-100K (Zhu et al., CVPR 2016) as a traffic sign classification task.

We use the HuggingFace copy `PrashantDixit0/TT-100K`. It has 2048x2048 street scenes. Every sign
that is at least 32 px wide becomes one sample. The sample is a 1008 px crop around the sign with
all other signs blacked out. The label is one of the 67 sign types with at least 30 instances.
We train on the HF `train` and `val` splits and evaluate on `test`.
"""

import io
import json
import pathlib
from collections import Counter

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from tqdm import tqdm

from data.augment import eval_transform, train_transform

HF_REPO = "PrashantDixit0/TT-100K"
NUM_CLASSES = 67
MIN_INSTANCES, MIN_SIGN_PX = 30, 32
AUGMENT = {
    "default": dict(scale=None, flip=True),
    "hard": dict(scale=None, flip=True, jitter=0.3, magnitude=9, erase=0.25),
}


def _splits(root):
    from datasets import concatenate_datasets, load_dataset
    hf = load_dataset(HF_REPO, cache_dir=str(root))
    return {"train": concatenate_datasets([hf["train"], hf["val"]]), "valid": hf["test"]}


def prepare(root):
    """Download the dataset into `root`. Nothing happens if it is already cached."""
    _splits(root)


def _objects(split):
    """Yield the list of (label, box) pairs of each image. The boxes are converted from xywh to xyxy."""
    for row in tqdm(split.select_columns(["objects"]), desc="TT-100K: indexing", leave=False):
        objects = row["objects"]
        if isinstance(objects, dict):  # HF stores a list of structs as a dict of lists.
            objects = [{"category": c, "bbox": b} for c, b in zip(objects["category"], objects["bbox"])]
        yield [(o["category"], (o["bbox"][0], o["bbox"][1], o["bbox"][0] + o["bbox"][2], o["bbox"][1] + o["bbox"][3]))
               for o in objects]


def _samples(split, vocab):
    """Return one sample for every large enough sign.

    A sample holds the image index, the target box, the class and the boxes of the other signs.
    """
    samples = []
    for index, objects in enumerate(_objects(split)):
        for i, (label, box) in enumerate(objects):
            if label in vocab and min(box[2] - box[0], box[3] - box[1]) >= MIN_SIGN_PX:
                others = [b for j, (_, b) in enumerate(objects) if j != i]
                samples.append((index, box, vocab[label], others))
    return samples


def _index(root, splits):
    """Return the class vocabulary and the samples of both splits. They are built once and cached in `root`."""
    path = pathlib.Path(root) / "index.json"
    if path.exists():
        return json.loads(path.read_text())
    counts = Counter(label for objects in _objects(splits["train"]) for label, _ in objects)
    vocab = {label: i for i, label in enumerate(sorted(l for l, n in counts.items() if n >= MIN_INSTANCES))}
    index = {"vocab": vocab, **{name: _samples(split, vocab) for name, split in splits.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index))
    return index


def _crop_around(image, box, size):
    """Crop a size x size window centered on `box`.

    The window is shifted to stay inside the image. It is zero-padded if the image is too small.
    """
    x0 = min(max(int((box[0] + box[2]) / 2 - size / 2), 0), max(image.width - size, 0))
    y0 = min(max(int((box[1] + box[3]) / 2 - size / 2), 0), max(image.height - size, 0))
    crop = image.crop((x0, y0, x0 + size, y0 + size))
    if crop.size == (size, size) and x0 + size <= image.width and y0 + size <= image.height:
        return crop
    padded = Image.new("RGB", (size, size))
    padded.paste(image.crop((x0, y0, min(x0 + size, image.width), min(y0 + size, image.height))))
    return padded


class TT100K(Dataset):
    def __init__(self, split, samples, img_size, transform):
        self.split, self.samples, self.img_size, self.transform = split, samples, img_size, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_index, box, label, others = self.samples[index]
        image = self.split[image_index]["image"]
        if isinstance(image, dict):  # The column holds raw bytes if it is not decoded.
            image = Image.open(io.BytesIO(image["bytes"]))
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for other in others:
            draw.rectangle(other, fill=(0, 0, 0))
        return self.transform(_crop_around(image, box, self.img_size)), label


def load(cfg, split):
    splits = _splits(cfg.data_root)
    if split not in splits:
        raise ValueError(f"TT-100K has no '{split}' split, expected 'train' or 'valid'.")
    samples = _index(cfg.data_root, splits)[split]
    # The crops already have size img_size, so evaluation needs no resize either.
    transform = train_transform(cfg.img_size, **AUGMENT[cfg.augment]) if split == "train" else eval_transform(cfg.img_size)
    return TT100K(splits[split], samples, cfg.img_size, transform)
