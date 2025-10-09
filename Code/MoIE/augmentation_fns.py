#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   augmentation_fns.py
@Time    :   2024/11/11 17:58:54
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   None
'''
import PIL.Image
import random
import numpy as np

from PIL import Image, ImageOps


def randomScaleCrop(img:PIL.Image.Image, mask:PIL.Image.Image, crop_size=384):
    # Scale
    w = int(random.uniform(0.5, 1.5) * img.size[0])
    h = int(random.uniform(0.5, 1.5) * img.size[1])
    img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

    # Crop
    w, h = img.size
    if w < crop_size or h < crop_size:
        padding = np.maximum(0, np.maximum((crop_size - w) // 2 + 5, (crop_size - h) // 2 + 5))
        img = ImageOps.expand(img, border=padding, fill=0)
        mask = ImageOps.expand(mask, border=padding, fill=0)

    assert img.width == mask.width
    assert img.height == mask.height
    w, h = img.size
    th, tw = crop_size, crop_size  # target size
    if w == tw and h == th:
        return img, mask

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    img = img.crop((x1, y1, x1 + tw, y1 + th))
    mask = mask.crop((x1, y1, x1 + tw, y1 + th))
    return img, mask

def randomRotate(img:PIL.Image.Image, mask:PIL.Image.Image):
    rotate_degree = random.randint(1, 4) * 90
    img = img.rotate(rotate_degree, Image.BILINEAR, expand=0)
    mask = mask.rotate(rotate_degree, Image.NEAREST, expand=0)

    return img, mask