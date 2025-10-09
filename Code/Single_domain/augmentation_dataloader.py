#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   augmentation_dataloader.py
@Time    :   2025/10/09 12:48:24
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   None
'''

import os
import logging
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from augmentation_fns import randomScaleCrop, randomRotate

# ---------------------- SeparateDataset Class ---------------------- #
class SeparateDataset(Dataset):
    """
    Dataset class for handling slices from a single site.
    Supports train, test, and valid modes along with optional data augmentation.
    """
    def __init__(self, npz_root_dir, mode):
        """
        Initialize the dataset.

        Args:
            npz_root_dir (str): Root directory containing npz files.
            mode (str): Dataset mode ('train', 'test', or 'valid').
        """
        self.npz_dir = os.path.join(npz_root_dir, mode)
        self.mode = mode
        self.slices = [file for file in os.listdir(self.npz_dir) if not file.startswith('.')]
        self.slices.sort()
        self.cases = list(np.unique([v.split('_')[0] for v in self.slices]))
        self.cases.sort()

    def __len__(self):
        return len(self.slices)

    def _data_augmentation(self, img, gt):
        """
        Perform random flip augmentation.

        Args:
            img (np.array): Image array.
            gt (np.array): Ground truth mask.

        Returns:
            tuple: Augmented image and ground truth.
        """
        if random.randint(0, 1) == 1:
            img = img[::-1, ...]
            gt = gt[::-1, ...]
        return img, gt

    def _case2slice(self):
        """
        Filter slices based on the selected cases.

        Returns:
            list: Filtered slice names.
        """
        reduced_slices = []
        for v in range(len(self.slices)):
            v_case = self.slices[v].split('_')[0]
            if v_case in self.cases:
                reduced_slices.append(self.slices[v])
        return reduced_slices

    def reduce(self, num: int):
        """
        Randomly reduce the number of cases in the dataset.

        Args:
            num (int): Number of cases to retain.
        """
        assert num <= len(self.cases)
        logging.info(f'Before reduce operation: {len(self.slices)} slices ({len(self.cases)} cases) in {self.npz_dir}')
        self.cases = random.sample(self.cases, num)
        self.cases.sort()
        self.slices = self._case2slice()
        logging.info(f'After reduce operation: {len(self.slices)} slices ({len(self.cases)} cases)')

    def sample(self, ind: list):
        """
        Select specific cases by their indices.

        Args:
            ind (list): Indices of cases to keep.
        """
        logging.info(f'Before sample operation: {len(self.slices)} slices ({len(self.cases)} cases) in {self.npz_dir}')
        self.cases = [self.cases[v] for v in ind]
        self.cases.sort()
        self.slices = self._case2slice()
        logging.info(f'After sample operation: {len(self.slices)} slices ({len(self.cases)} cases)')
        return self

    def to_multilabel(self, pre_mask, classes=4):
        """
        Convert a single-channel mask to multi-channel.

        Args:
            pre_mask (np.array): Input binary or integer mask.
            classes (int): Number of classes.

        Returns:
            np.array: Multi-channel mask.
        """
        mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes), dtype=np.float32)
        for i in range(classes):
            mask[:, :, i] = (pre_mask == i)
        return mask

    def __getitem__(self, item):
        """
        Get a single sample from the dataset.

        Args:
            item (int): Index of the sample.

        Returns:
            dict: Dictionary containing slice name, image, and ground truth mask.
        """
        slice_name = os.path.join(self.npz_dir, self.slices[item])
        data = np.load(slice_name)
        img, gt = data['arr_0'], data['arr_1']

        # Perform data augmentation for training mode
        if self.mode == 'train':
            img, gt = self._data_augmentation(img, gt)

        # Normalize and preprocess image
        gt = gt.astype(np.uint8)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        image = Image.fromarray(img).convert('L')
        gt = Image.fromarray(gt).convert('L')
        shape = image.size[0]

        if self.mode == 'train':
            if random.random() > 0.5:
                image, gt = randomScaleCrop(image, gt, shape)
            if random.random() > 0.5:
                image, gt = randomRotate(image, gt)
            if random.random() > 0.5:
                image, gt = image.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image, gt = image.transpose(Image.FLIP_TOP_BOTTOM), gt.transpose(Image.FLIP_TOP_BOTTOM)

        gt = np.array(gt)
        image = np.array(image).astype(np.float32) / np.max(image.astype(np.float32))
        image = (image - 0.5) / 0.5
        img = np.repeat(image[np.newaxis, :, :], 3, axis=0)
        gt = self.to_multilabel(gt, classes=3)
        gt = gt.transpose([2, 0, 1])

        return {'slice_name': self.slices[item], 'img': torch.from_numpy(img), 'gt': torch.from_numpy(gt)}

# ---------------------- JointDataset Class ---------------------- #
class JointDataset(Dataset):
    """
    Dataset class for handling slices from multiple sites.
    """
    def __init__(self, npz_root_dir, sites, mode, percent=1.0):
        """
        Initialize the joint dataset.

        Args:
            npz_root_dir (str): Root directory containing data for all sites.
            sites (list): List of site names to include in the dataset.
            mode (str): Dataset mode ('train' or 'test').
            percent (float): Percentage of data to use (default is 100%).
        """
        self.slices = []
        self.mode = mode
        self.sites = sites
        self.npz_root_dir = npz_root_dir
        for site in sites:
            npz_dir = os.path.join(npz_root_dir, site, mode)
            self.slices += [os.path.join(npz_dir, file) for file in os.listdir(npz_dir) if not file.startswith('.')]
        self.slices.sort()

        # Select a subset of slices if percent < 1.0
        if 0 < percent < 1.0:
            num = int(len(self.slices) * percent)
            start = random.randint(0, len(self.slices) - num)
            self.slices = self.slices[start:start + num]
        print(f'Created {self.mode} dataset from {self.npz_root_dir} ({sites}) with {len(self.slices)} slices.')

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):
        """
        Get a single sample from the dataset.

        Args:
            item (int): Index of the sample.

        Returns:
            dict: Dictionary containing site name, slice name, image, and ground truth mask.
        """
        slice_name = self.slices[item]
        site_name = slice_name.split("/")[-3]
        data = np.load(slice_name)
        img, gt = data['arr_0'], data['arr_1']

        # Perform data augmentation for training mode
        if self.mode == 'train':
            img, gt = self._data_augmentation(img, gt)

        gt = gt.astype(np.uint8)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        image = Image.fromarray(img).convert('L')
        gt = Image.fromarray(gt).convert('L')
        shape = image.size[0]

        if self.mode == 'train':
            if random.random() > 0.5:
                image, gt = randomScaleCrop(image, gt, shape)
            if random.random() > 0.5:
                image, gt = randomRotate(image, gt)

        gt = np.array(gt)
        image = np.array(image).astype(np.float32) / np.max(image.astype(np.float32))
        image = (image - 0.5) / 0.5
        img = np.repeat(image[np.newaxis, :, :], 3, axis=0)
        gt = self.to_multilabel(gt, classes=3)
        gt = gt.transpose([2, 0, 1])

        return {'site_name': site_name, 'slice_name': slice_name, 'img': torch.from_numpy(img), 'gt': torch.from_numpy(gt)}

# ---------------------- DataLoader Utilities ---------------------- #
def offline_loader(dataroot, sites, batch_size):
    """
    Create DataLoaders for offline training.

    Args:
        dataroot (str): Root directory of the dataset.
        sites (list): List of site names.
        batch_size (int): Batch size for training.

    Returns:
        tuple: (train_loader, test_loaders, sites)
    """
    train_dataset = JointDataset(npz_root_dir=dataroot, sites=sites, mode='train')
    test_datasets = {site: SeparateDataset(npz_root_dir=os.path.join(dataroot, site), mode='test') for site in sites}
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    test_loaders = {site: DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
                    for site, dataset in test_datasets.items()}
    return train_loader, test_loaders, train_dataset.get_sites()