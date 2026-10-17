"""Image transforms and CutMix shared by all datasets."""

import random

import torch
import torch.nn.functional as F
from PIL import ImageFilter
from timm.data.transforms import RandomResizedCropAndInterpolation
from torch.utils.data import default_collate
from torchvision import transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def eval_transform(img_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(img_size, scale=(1.0, 1.0), flip=False, jitter=0.0, magnitude=5, erase=0.0):
    """Build the training transform."""
    steps: list = []
    if scale is not None:
        steps.append(RandomResizedCropAndInterpolation(img_size, scale=scale, interpolation="bicubic"))
    if flip:
        steps.append(transforms.RandomHorizontalFlip())
    if jitter:
        steps.append(transforms.ColorJitter(jitter, jitter, jitter, hue=0.1))
    steps += [transforms.RandAugment(2, magnitude), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if erase:
        steps.append(transforms.RandomErasing(p=erase, scale=(0.02, 0.2)))
    return transforms.Compose(steps)


class _Blur:
    """Gaussian blur."""
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def three_augment(img_size):
    """3-Augment (Touvron et al., DeiT III).
    """
    return transforms.Compose([
        RandomResizedCropAndInterpolation(img_size, scale=(0.08, 1.0), interpolation="bicubic"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([transforms.Grayscale(3), transforms.RandomSolarize(128, p=1.0), _Blur()]),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CutMix:
    """Batch collate function that applies CutMix and label smoothing.
    """

    def __init__(self, num_classes, alpha=1.0, smoothing=0.0):
        self.num_classes, self.alpha, self.smoothing = num_classes, alpha, smoothing

    def __call__(self, batch):
        images, labels = default_collate(batch)
        targets = F.one_hot(labels, self.num_classes).float()
        targets = targets * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            perm = torch.randperm(images.shape[0])
            h, w = images.shape[-2:]
            cut_h, cut_w = int(h * (1.0 - lam) ** 0.5), int(w * (1.0 - lam) ** 0.5)
            cy, cx = random.randrange(h), random.randrange(w)
            y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
            x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)  # The box can be clipped at the image border. Use its real area.
            targets = lam * targets + (1.0 - lam) * targets[perm]
        return images, targets
