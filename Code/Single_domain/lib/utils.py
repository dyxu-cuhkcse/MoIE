#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/11/11 18:01:09
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   Utility functions for data loading, contour visualization, and segmentation overlay.
'''

import os
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt


def batch_from_iterator(iterator, dataloader):
    """
    Fetch a batch from an iterator wrapped around a DataLoader.
    If the iterator is depleted, reset it and fetch a new batch.

    Args:
        iterator: Iterator object for the DataLoader.
        dataloader: PyTorch DataLoader instance.

    Returns:
        tuple: (images, ground truths, updated iterator)
    """
    try:
        batch = next(iterator)
    except StopIteration:
        # Reset the iterator if it is exhausted
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch['img'].cuda(), batch['gt'].cuda(), iterator


def check_folder(log_dir):
    """
    Check if a folder exists, and create it if it does not.

    Args:
        log_dir (str): Path to the directory.

    Returns:
        str: Path to the directory.
    """
    if not os.path.exists(log_dir):
        print(f"Allocating '{log_dir}'")
        os.makedirs(log_dir)
    return log_dir


def add_countor(In, Seg, Color=(0, 255, 0)):
    """
    Overlay the segmentation contour onto an image.

    Args:
        In (PIL.Image): Input image.
        Seg (PIL.Image): Segmentation mask.
        Color (tuple): RGB color for the contour.

    Returns:
        PIL.Image: Image with segmentation contour overlay.
    """
    Out = In.copy()
    [H, W] = In.size

    for i in range(H):
        for j in range(W):
            if (i == 0 or i == H - 1 or j == 0 or j == W - 1) and Seg.getpixel((i, j)) != 0:
                Out.putpixel((i, j), Color)
            elif (
                Seg.getpixel((i, j)) != 0 and
                not (
                    Seg.getpixel((i - 1, j)) != 0 and
                    Seg.getpixel((i + 1, j)) != 0 and
                    Seg.getpixel((i, j - 1)) != 0 and
                    Seg.getpixel((i, j + 1)) != 0
                )
            ):
                Out.putpixel((i, j), Color)
    return Out


def add_segmentation(image, seg_name, Color=(0, 255, 0)):
    """
    Add segmentation contours to an image.

    Args:
        image (PIL.Image): Input image.
        seg_name (str): Path to the segmentation mask file.
        Color (tuple): RGB color for the contour.

    Returns:
        PIL.Image: Image with segmentation overlay.
    """
    seg = Image.open(seg_name).convert('L')
    seg = np.asarray(seg)

    # Resize the segmentation mask if the dimensions do not match
    if image.size[1] != seg.shape[0] or image.size[0] != seg.shape[1]:
        print("Segmentation mask has been resized")
    
    # Perform morphological operations to clean the segmentation mask
    strt = ndimage.generate_binary_structure(2, 1)
    seg = np.asarray(ndimage.morphology.binary_opening(seg, strt), np.uint8)
    seg = np.asarray(ndimage.morphology.binary_closing(seg, strt), np.uint8)

    # Add segmentation contours to the image
    img_show = add_countor(image, Image.fromarray(seg), Color)

    # Perform dilation to emphasize the contours
    seg = np.asarray(ndimage.morphology.binary_dilation(seg, strt), np.uint8)
    img_show = add_countor(img_show, Image.fromarray(seg), Color)
    return img_show


def show_seg_contour(img_name, save_name, seg_name, seg_name2, gld_name):
    """
    Visualize and save an image with segmentation contours overlaid.

    Args:
        img_name (str): Path to the input image.
        save_name (str): Path to save the resulting image.
        seg_name (str): Path to the primary segmentation mask.
        seg_name2 (str): Path to the secondary segmentation mask.
        gld_name (str): Path to the ground truth mask.
    """
    img = Image.open(img_name)

    if gld_name is not None and seg_name is not None:
        img_show = add_segmentation(img, gld_name, Color=(255, 255, 0))  # Ground truth (yellow)
        img_show = add_segmentation(img_show, seg_name2, Color=(255, 0, 0))  # Secondary segmentation (red)
        img_show = add_segmentation(img_show, seg_name, Color=(0, 255, 0))  # Primary segmentation (green)
    elif seg_name is None and gld_name is not None:
        img_show = add_segmentation(img, gld_name, Color=(255, 255, 0))  # Ground truth only (yellow)
    elif seg_name is not None and gld_name is None:
        img_show = add_segmentation(img, seg_name, Color=(0, 255, 0))  # Segmentation only (green)

    # Display and save the visualization
    plt.imshow(img_show)
    plt.axis('off')
    plt.show()
    img_show.save(save_name)