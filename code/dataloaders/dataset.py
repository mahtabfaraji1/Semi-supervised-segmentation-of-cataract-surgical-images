import itertools
import os
import random

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


# Dataset

class BaseDataSets(Dataset):
    """
    Dataset for cataract surgical image segmentation supporting mixed
    labeled and unlabeled samples in a single directory.

    Args:
        base_dir:   Root path containing train/val subdirectories.
        split:      'train' or 'val'.
        num:        If set, truncate training set to this many samples.
        transform:  Transform applied to every sample (e.g. RandomGenerator).
    """

    def __init__(self, base_dir, split="train", num=None, transform=None):
        self._base_dir = base_dir
        self.split     = split
        self.transform = transform

        images_dir = os.path.join(self._base_dir, split, 'images')
        labels_dir = os.path.join(self._base_dir, split, 'labels')

        self.image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith('.png')
        ])
        # Build a set of label filenames for O(1) lookup
        self.label_paths = sorted([
            os.path.join(labels_dir, f)
            for f in os.listdir(labels_dir)
            if f.endswith('.png')
        ]) if os.path.isdir(labels_dir) else []

        self._label_name_set = set(
            os.path.basename(p) for p in self.label_paths
        )

        if num is not None and split == "train":
            self.image_paths = self.image_paths[:num]

        print(f"[{split}] {len(self.image_paths)} images, "
              f"{len(self.label_paths)} labeled.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)   # BGR uint8, shape (H, W, 3)
        image = np.array(image, dtype=np.uint8)

        img_filename = os.path.basename(image_path)
        if img_filename in self._label_name_set:
            # Labeled sample: load grayscale mask (pixel = class index 0-4)
            label_path = os.path.join(
                self._base_dir, self.split, 'labels', img_filename
            )
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = np.array(label, dtype=np.uint8)
        else:
            # Unlabeled sample: zero mask (not used in supervised loss)
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)

        sample["idx"] = idx
        return sample

    def get_image_names(self, idxs):
        """Return image filenames for the given list of indices."""
        return [os.path.basename(self.image_paths[i]) for i in idxs]


# Augmentation transforms

def random_rot_flip(image, label=None):

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    if label is not None:
        label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.flip(label, axis=axis).copy()

    return image, label



class RandomGenerator(object):


    def __init__(self, output_size):
        self.output_size = output_size  # [H, W], paper: [256, 256]

    def __call__(self, sample):
        image, label = sample["image"], sample.get("label", None)


        image, label = random_rot_flip(image, label)

        x, y, _ = image.shape
        image = zoom(
            image,
            (self.output_size[0] / x, self.output_size[1] / y, 1),
            order=0
        )
        if label is not None:
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0
            )

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        if label is not None:
            label = torch.from_numpy(label.astype(np.uint8))

        return {"image": image, "label": label}


# Two-stream batch sampler

class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices,
                 batch_size, secondary_batch_size):
        self.primary_indices      = primary_indices
        self.secondary_indices    = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size   = batch_size - secondary_batch_size

        assert len(self.primary_indices)   >= self.primary_batch_size   > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter   = _iterate_once(self.primary_indices)
        secondary_iter = _iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for primary_batch, secondary_batch in zip(
                _grouper(primary_iter,   self.primary_batch_size),
                _grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def _iterate_once(iterable):
    return np.random.permutation(iterable)


def _iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def _grouper(iterable, n):
    """Collect data into fixed-length chunks: grouper('ABCDE', 3) -> ABC DE."""
    args = [iter(iterable)] * n
    return zip(*args)