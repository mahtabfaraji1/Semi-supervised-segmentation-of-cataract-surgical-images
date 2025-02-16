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

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # Get image and label paths based on the split (train or val)
        self.image_paths = sorted(glob(os.path.join(self._base_dir, split, 'images', '*.png')))
        self.label_paths = sorted(glob(os.path.join(self._base_dir, split, 'labels', '*.png')))

        # Ensure the number of images matches the number of labels
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"

        if num is not None and self.split == "train":
            self.image_paths = self.image_paths[:num]
            self.label_paths = self.label_paths[:num]

        print(f"Total {len(self.image_paths)} samples in {self.split} set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load the image and label
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Convert images and labels to numpy arrays
        image = np.array(image)
        label = np.array(label)

        sample = {"image": image, "label": label}

        if self.transform:
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        sample["idx"] = idx
        return sample

    def get_image_names(self, idxs):
        """Return the image file names for the given indices."""
        return [os.path.basename(self.image_paths[idx]) for idx in idxs]


class BaseDataSetsGray(Dataset):
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

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # Get image and label paths based on the split (train or val)
        self.image_paths = sorted(glob(os.path.join(self._base_dir, split, 'images', '*.png')))
        self.label_paths = sorted(glob(os.path.join(self._base_dir, split, 'labels', '*.png')))

        # Ensure the number of images matches the number of labels
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"

        if num is not None and self.split == "train":
            self.image_paths = self.image_paths[:num]
            self.label_paths = self.label_paths[:num]

        print(f"Total {len(self.image_paths)} samples in {self.split} set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load the image and label
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Convert images and labels to numpy arrays
        image = np.array(image)
        label = np.array(label)

        sample = {"image": image, "label": label}

        if self.transform:
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

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
    image = np.rot90(image, k)  # Rotate the image by 0, 90, 180, or 270 degrees

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
        image, label = sample["image"], sample.get("label", None)  # Ensure label can be None

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)  # Apply rotation and flip (label may be None)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)  # Apply random rotation (label may be None)

        # Resize the image and label
        if len(image.shape) == 3:
            x, y, _ = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if label is not None:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = my_PreProc(image)
        # Convert image and label to PyTorch tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        if label is not None:
            label = torch.from_numpy(label.astype(np.uint8))

        sample = {"image": image, "label": label}
        return sample


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample["image"], sample.get("label", None)  # Default to None if no label
#
#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)  # Apply rotation
#
#         x, y, _ = image.shape
#         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
#
#         if label is not None:
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         if label is not None:
#             label = torch.from_numpy(label.astype(np.uint8))
#
#         sample = {"image": image, "label": label}
#         return sample


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample["image"], sample["label"]
#         # ind = random.randrange(0, img.shape[0])
#         # image = img[ind, ...]
#         # label = lab[ind, ...]
#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         # image = color.rgb2gray(image)
#         x, y,_= image.shape
#         # print("image.shape",image.shape)
#         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=0)
#         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.uint8))
#         sample = {"image": image, "label": label}
#         return sample


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
        x, y, _ = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)


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


import numpy as np
import cv2

#My pre processing (use for both training and testing!)
def my_PreProc(data):
    # assert(len(data.shape)==3)
    # assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs

#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#convert RGB image in black and white
def rgb2gray(rgb):
    # assert (len(rgb.shape)==4)  #4D arrays
    # assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    # bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#==== histogram equalization
def histo_equalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(img):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img_equalized = np.empty(img.shape)
    img_equalized = clahe.apply(np.array(img, dtype=np.uint8))
    # for i in range(imgs.shape[0]):
    #     imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return img_equalized


# ===== normalize over the dataset
def dataset_normalized(img):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(img.shape)
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img-img_mean)/img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (
                np.max(img_normalized) - np.min(img_normalized))) * 255
    # for i in range(imgs.shape[0]):
    #     imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return img_normalized


def adjust_gamma(img, gamma=1.0):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    # new_img = np.empty(img.shape)
    new_img = cv2.LUT(np.array(img, dtype=np.uint8), table)
    # for i in range(imgs.shape[0]):
    #     new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_img

