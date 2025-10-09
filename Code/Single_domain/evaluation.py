#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2025/01/25 14:38:24
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   Evaluation utilities for model performance
'''

import torch
import numpy as np
from PIL import Image
from monai.networks.utils import one_hot

def visualize_channels(tensor, save_prefix="channel"):
    """
    Visualize each channel of a tensor as separate PNG images.

    Args:
        tensor (torch.Tensor): Input tensor of shape [1, 3, 256, 256].
        save_prefix (str): Prefix for the saved file names.
    """
    # Ensure the tensor is on the CPU and convert it to a NumPy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.squeeze(0)  # Remove batch dimension, shape becomes [3, 256, 256]
    tensor = tensor.numpy()
    
    # Loop through each channel
    for i in range(3):
        channel = tensor[i]  # Extract channel [256, 256]

        # Normalize the channel values to the range [0, 255]
        if channel.max() > 1 or channel.min() < 0:
            channel = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
        elif channel.max() <= 1:  # If values are already in [0, 1]
            channel = (channel * 255).astype(np.uint8)
        
        # Save the channel as a PNG image
        img = Image.fromarray(channel)
        img.save(f"./visualization/{save_prefix}_channel_{i}.png")
        print(f"Saved {save_prefix}_channel_{i}.png")

def calculate_dice(pred, gt):
    """
    Calculate Dice similarity coefficient (DSC) for each class and the mean Dice.

    Args:
        pred (torch.Tensor): Predicted segmentation, shape [1, 3, 256, 256].
        gt (torch.Tensor): Ground truth segmentation, shape [1, 3, 256, 256].

    Returns:
        tuple: (list of Dice scores for each class, mean Dice score).
    """
    # Convert predictions and ground truth to class labels
    pred = torch.argmax(pred, dim=1)  # Shape becomes [1, 256, 256]
    gt = torch.argmax(gt, dim=1)      # Shape becomes [1, 256, 256]
    
    dices = []  # Store Dice scores for each class
    for class_id in range(2):  # Evaluate for classes 0 and 1 (assumes binary segmentation)
        pred_class = (pred == class_id)  # Binary mask for the predicted class
        gt_class = (gt == class_id)      # Binary mask for the ground truth class

        # Calculate intersection and union
        intersection = torch.sum(pred_class * gt_class)
        union = torch.sum(pred_class) + torch.sum(gt_class)

        # Handle edge case: if both prediction and ground truth are empty
        if union == 0:
            dice = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
        else:
            dice = (2.0 * intersection) / union

        dices.append(dice.item())
    
    # Compute mean Dice score
    mean_dice = sum(dices) / len(dices)
    return dices, mean_dice

def evaluation_case(model, dataloader, site_name):
    """
    Evaluate a model on a single test dataset.

    Args:
        model (nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        site_name (str): Name of the test site.

    Returns:
        float: Mean Dice score for the dataset.
    """
    all_dices = []  # Store Dice scores for all batches
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Iterate through the test dataset
        for batch in dataloader:
            name, img, gt = batch['slice_name'], batch['img'].cuda(), batch['gt'].cuda()
            
            # Ensure batch size is 1 for testing
            assert len(name) == 1 and img.shape[0] == 1 and gt.shape[0] == 1
            
            # Run the forward pass
            result = model(img)

            # Convert predictions to one-hot encoding
            pred = one_hot(torch.argmax(result.detach(), dim=1, keepdim=True), num_classes=3, dim=1)  # Shape: [1, 1, 256, 256]

            # Calculate Dice scores for the current batch
            class_dices, avg_dice = calculate_dice(pred, gt)
            all_dices.append(avg_dice)  # Store the mean Dice score for this batch

        # Compute and return the mean Dice score across all batches
        return sum(all_dices) / len(all_dices)