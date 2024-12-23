import os
import cv2
import torch
from skimage import color
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image


import os
from torch.utils.data import Dataset
from PIL import Image
# class BaseDataSets(Dataset):
#     def __init__(
#         self,
#         base_dir=None,
#         split="train",
#         num=None,
#         transform=None,
#         ops_weak=None,
#         ops_strong=None,
#     ):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.transform = transform
#         self.ops_weak = ops_weak
#         self.ops_strong = ops_strong
#
#         assert bool(ops_weak) == bool(
#             ops_strong
#         ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"
#
#         if self.split == "train":
#             with open(self._base_dir + "/train_slices.list", "r") as f1:
#                 self.sample_list = f1.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list]
#
#         elif self.split == "val":
#             with open(self._base_dir + "/val.list", "r") as f:
#                 self.sample_list = f.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list]
#         if num is not None and self.split == "train":
#             self.sample_list = self.sample_list[:num]
#         print("total {} samples".format(len(self.sample_list)))
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         if self.split == "train":
#             h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
#         else:
#             h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
#         image = h5f["image"][:]
#         label = h5f["label"][:]
#         sample = {"image": image, "label": label}
#         if self.split == "train":
#             if None not in (self.ops_weak, self.ops_strong):
#                 sample = self.transform(sample, self.ops_weak, self.ops_strong)
#             else:
#                 sample = self.transform(sample)
#         sample["idx"] = idx
#         return sample


class BaseDataSetsIncludesUnlabeled(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=None, unlabeled=False):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.unlabeled = unlabeled  # New parameter to specify if this is an unlabeled dataset

        # Get image paths


        # For labeled data, also load label paths
        if not self.unlabeled:
            self.image_paths = sorted(glob(os.path.join(self._base_dir, split, 'images','labeled', '*.png')))
            self.label_paths = sorted(glob(os.path.join(self._base_dir, split, 'labels', '*.png')))
            assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"
        else:
            self.image_paths = sorted(glob(os.path.join(self._base_dir, split,  'images','unlabeled', '*.png')))
            self.label_paths = [None] * len(self.image_paths)  # No labels for unlabeled data

        if num is not None and self.split == "train":
            self.image_paths = self.image_paths[:num]
            if not self.unlabeled:
                self.label_paths = self.label_paths[:num]

        print(f"Total {len(self.image_paths)} samples in {self.split} set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        # Check if this is labeled data
        if not self.unlabeled:
            label_path = self.label_paths[idx]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Read the label
        else:
            # If it's unlabeled, create a zero-filled label with the same size as the image
            label = np.zeros((image.shape[0], image.shape[1]),
                             dtype=np.uint8)  # Zero image with the same height and width as the image

        # Apply any transformations (if provided)
        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)
        sample["image_name"] = os.path.basename(image_path)
        sample["idx"] = idx
        return sample

    def get_image_names(self, idxs):
        """Return the image file names for the given indices."""
        return [os.path.basename(self.image_paths[idx]) for idx in idxs]


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        # assert bool(ops_weak) == bool(
        #     ops_strong
        # ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # Get image and label paths based on the split (train or val)
        self.image_paths = sorted(glob(os.path.join(self._base_dir, split, 'images', '*.png')))
        self.label_paths = sorted(glob(os.path.join(self._base_dir, split, 'labels', '*.png')))

        # Ensure the number of images matches the number of labels
        # assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"

        # if num is not None and self.split == "train":
        #     self.image_paths = self.image_paths[:num]
        #     self.label_paths = self.label_paths[:num]

        print(f"Total {len(self.image_paths)} samples in {self.split} set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = np.array(image, dtype=np.float32)
        image = image /255.0
        label_path = image_path.replace('images', 'labels')

        if label_path in self.label_paths:
            # label_path = self.label_paths[idx]
            # Load the image and label

            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            # Convert images and labels to numpy arrays

            label = np.array(label)
        else:
            label = np.zeros((image.shape[0], image.shape[1]))
            label = np.array(label)
        sample = {"image": image, "label": label, "ops_strong": self.ops_strong}

        if self.transform:
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        # final_sample = self.ops_strong(sample)
        sample["idx"] = idx
        return sample

    def get_image_names(self, idxs):
        """Return the image file names for the given indices."""
        return [os.path.basename(self.image_paths[idx]) for idx in idxs]

def random_rot_flip(image, label=None):
    """
    Randomly rotates and flips an image, and also applies the same transformation to the label if it exists.
    """
    k = np.random.randint(0, 4)  # Randomly choose a number between 0 and 3 for 90-degree rotations
    image = np.rot90(image, k)   # Rotate the image by 0, 90, 180, or 270 degrees

    axis = np.random.randint(0, 2)  # Randomly choose whether to flip vertically or horizontally
    image = np.flip(image, axis=axis).copy()

    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()

    return image, label  # Always return both image and label (label may be None)


# def random_rot_flip(image, label=None):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     if label is not None:
#         label = np.rot90(label, k)
#         label = np.flip(label, axis=axis).copy()
#         return image, label
#     else:
#         return image


# def random_rotate(image, label):
#     angle = np.random.randint(-20, 20)
#     image = ndimage.rotate(image, angle, order=0, reshape=False)
#     label = ndimage.rotate(label, angle, order=0, reshape=False)
#     return image, label

def random_rotate(image, label=None):
    """Randomly rotate both the image and the label (if available)."""
    angle = np.random.randint(-20, 20)  # Random rotation angle

    # Rotate the image
    image = ndimage.rotate(image, angle, order=0, reshape=False)

    # Only rotate the label if it exists
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)

    return image, label

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y, _ = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)

        # return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_raw, label_raw, ops_strong_flag= sample["image"], sample.get("label", None), sample["ops_strong"]  # Ensure label can be None

        if random.random() > 0.5:
            image_raw, label_raw = random_rot_flip(image_raw, label_raw)  # Apply rotation and flip (label may be None)
        elif random.random() > 0.5:
            image_raw, label_raw = random_rotate(image_raw, label_raw)  # Apply random rotation (label may be None)

        # Resize the image and label
        x, y, _ = image_raw.shape
        image_raw = zoom(image_raw, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)

        if label_raw is not None:
            label_raw = zoom(label_raw, (self.output_size[0] / x, self.output_size[1] / y), order=0)


        if label_raw is not None:
            label_raw = torch.from_numpy(label_raw.astype(np.uint8))

        if ops_strong_flag:
            image_strong = color_jitter(image_raw).type("torch.FloatTensor")
            image_strong = image_strong.permute([1, 2, 0])
            image_strong = image_strong.unsqueeze(0)
        # Convert image and label to PyTorch tensors
        image_raw = torch.from_numpy(image_raw.astype(np.float32)).unsqueeze(0)
        # image_strong = image_strong.unsqueeze(0)

        sample = {
            "image": image_raw,
            # "image_weak": image_weak,
            "image_ema": image_strong,
            "label": label_raw,
        }

        # sample = {"image": image, "label": label}
        return sample

class StrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample.get("label", None)  # Ensure label can be None
        # image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        # image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            # "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y,_ = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=0)
class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y,_ = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
