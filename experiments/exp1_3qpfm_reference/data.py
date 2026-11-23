"""FashionIQ loaders matching QuRe's caption composition, without negatives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


FASHION_TYPES = ("dress", "shirt", "toptee")


def compose_caption(captions: Sequence[str]) -> str:
    """The exact relative-caption composition used by QuRe FashionIQDataset."""
    return captions[0].strip(".?, ").capitalize() + " and " + captions[1].strip(".?, ")


class FashionIQPositiveTrain(Dataset):
    def __init__(self, root: str | Path, transform, dress_types=FASHION_TYPES):
        self.root = Path(root)
        self.transform = transform
        self.triplets = []
        for dress_type in dress_types:
            with (self.root / "captions" / f"cap.{dress_type}.train.json").open() as stream:
                self.triplets.extend(json.load(stream))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        item = self.triplets[index]
        reference = self.transform(Image.open(self.root / "images" / f"{item['candidate']}.png").convert("RGB"))
        target = self.transform(Image.open(self.root / "images" / f"{item['target']}.png").convert("RGB"))
        # Keep target name so the loss can treat repeated target images inside
        # one batch as multiple positives rather than false negatives.
        return reference, target, compose_caption(item["captions"]), item["target"]


class FashionIQGallery(Dataset):
    def __init__(self, root: str | Path, dress_type: str, transform):
        self.root, self.transform = Path(root), transform
        with (self.root / "image_splits" / f"split.{dress_type}.val.json").open() as stream:
            self.names = json.load(stream)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        return self.transform(Image.open(self.root / "images" / f"{name}.png").convert("RGB")), name


class FashionIQQueries(Dataset):
    def __init__(self, root: str | Path, dress_type: str, transform):
        self.root, self.transform = Path(root), transform
        with (self.root / "captions" / f"cap.{dress_type}.val.json").open() as stream:
            self.triplets = json.load(stream)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        item = self.triplets[index]
        reference = self.transform(Image.open(self.root / "images" / f"{item['candidate']}.png").convert("RGB"))
        return reference, compose_caption(item["captions"]), item["target"]


def collate_train(batch):
    references, targets, captions, target_names = zip(*batch)
    return torch.stack(references), torch.stack(targets), list(captions), list(target_names)


def collate_gallery(batch):
    images, names = zip(*batch)
    return torch.stack(images), list(names)


def collate_queries(batch):
    references, captions, targets = zip(*batch)
    return torch.stack(references), list(captions), list(targets)


def make_train_loader(root, transform, batch_size, num_workers):
    return DataLoader(
        FashionIQPositiveTrain(root, transform), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_train,
    )


def make_validation_loaders(root, transform, batch_size, num_workers):
    result = {}
    for dress_type in FASHION_TYPES:
        result[dress_type] = {
            "gallery": DataLoader(FashionIQGallery(root, dress_type, transform), batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_gallery),
            "queries": DataLoader(FashionIQQueries(root, dress_type, transform), batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_queries),
        }
    return result
