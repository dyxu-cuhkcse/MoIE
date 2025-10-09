#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   souce_std_mean.py
@Time    :   2024/12/17 12:31:36
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   Calculate running statistics (mean and std) for intermediate layers of a pretrained UNet model on a specific dataset.
'''

import os
import random
import torch
import argparse
import numpy as np
from tqdm import tqdm
import Unet
from augmentation_dataloader import offline_loader


def calculate_running_statistics(test_loader, pretrained_model):
    """
    Calculate running mean and standard deviation for each block in a pretrained UNet.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        pretrained_model (nn.Module): Pretrained UNet model.

    Returns:
        dict: Dictionary containing mean and std for each block ('conv1', 'conv2', 'conv3', 'conv4').
    """
    pretrained_model.eval()

    # Define blocks and corresponding channels
    blocks = ['conv1', 'conv2', 'conv3', 'conv4']
    channels = [32, 64, 128, 256]
    statistics = {}
    
    # Initialize statistics for each block
    for i, block in enumerate(blocks):
        h = w = 256 // (2 ** i)  # Downsampled spatial size
        statistics[block] = {
            'sum': torch.zeros(channels[i], h, w).cuda(),   # Sum for mean calculation
            'sum_squared': torch.zeros(channels[i], h, w).cuda(),  # Sum of squares for variance
            'count': 0  # Total number of samples
        }

    # Extend UNet to return intermediate outputs
    class UnetWithIntermediates(pretrained_model.__class__):
        def forward(self, x):
            conv1 = self.conv1(x)
            pool1 = self.pool1(conv1)
            conv2 = self.conv2(pool1)
            pool2 = self.pool2(conv2)
            conv3 = self.conv3(pool2)
            pool3 = self.pool3(conv3)
            conv4 = self.conv4(pool3)
            pool4 = self.pool4(conv4)
            center = self.center(pool4)

            up_4 = self.up4(conv4, center)
            up_3 = self.up3(conv3, up_4)
            up_2 = self.up2(conv2, up_3)
            up_1 = self.up1(conv1, up_2)

            out = self.final(up_1)
            
            return out, {
                'conv1': conv1,
                'conv2': conv2,
                'conv3': conv3,
                'conv4': conv4
            }

    # Override model's forward method
    pretrained_model.__class__ = UnetWithIntermediates

    # Calculate statistics for each batch
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            img = batch['img'].cuda()

            # Forward pass to get outputs and intermediate feature maps
            _, intermediates = pretrained_model(img)

            # Update statistics for each block
            for block in blocks:
                features = intermediates[block]  # [B, C, H, W]
                batch_sum = features.sum(dim=0)  # Sum over batch dimension
                batch_sum_squared = (features ** 2).sum(dim=0)  # Sum of squares

                statistics[block]['sum'] += batch_sum
                statistics[block]['sum_squared'] += batch_sum_squared
                statistics[block]['count'] += features.size(0)  # Add batch size to count

            # Free memory
            del intermediates
            torch.cuda.empty_cache()

    # Compute final mean and standard deviation for each block
    results = {}
    for block in blocks:
        count = statistics[block]['count']
        mean = statistics[block]['sum'] / count  # Mean
        mean_squared = statistics[block]['sum_squared'] / count  # Mean of squares
        var = mean_squared - mean ** 2  # Variance
        std = torch.sqrt(torch.clamp(var, min=0))  # Standard deviation

        results[block] = {
            'mean': mean.cpu().numpy(),  # Convert to numpy
            'std': std.cpu().numpy()
        }

    return results


def save_statistics(stats, save_path):
    """
    Save computed statistics to a .npy file.

    Args:
        stats (dict): Computed statistics for each block.
        save_path (str): Path to save the .npy file.
    """
    save_dict = {}
    for block in ['conv1', 'conv2', 'conv3', 'conv4']:
        save_dict[f'{block}_mean'] = stats[block]['mean']
        save_dict[f'{block}_std'] = stats[block]['std']

    np.save(save_path, save_dict)
    print(f"Statistics saved to {save_path}")


def parse_args():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    desc = "PyTorch implementation of running statistics computation for pretrained UNet"
    parser = argparse.ArgumentParser(description=desc)

    # Arguments for model and data
    parser.add_argument('--gpu', type=str, default='0', help='GPU device number')
    parser.add_argument('--ckpt', type=str, default='./exp/debug', help='Path to pretrained model checkpoint')
    parser.add_argument('--data_npz', type=str, default='./data/fundus', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--network', type=str, default='Unet', help='Type of network (default: Unet)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    """
    Main function to calculate and save running statistics for a pretrained UNet.
    """
    args = parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load pretrained model
    if args.network == 'Unet':
        pretrained_model = Unet.Unet(num_channels=3, num_classes=3).cuda()
    model_state = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    pretrained_model.load_state_dict(model_state)

    # Load test data
    sites = ['BinRushed']
    _, all_test_loader, _ = offline_loader(args.data_npz, sites, args.batch_size)

    # Calculate statistics
    stats = calculate_running_statistics(all_test_loader['BinRushed'], pretrained_model)

    # Display results
    for block in ['conv1', 'conv2', 'conv3', 'conv4']:
        print(f"\n{block} statistics:")
        print(f"Mean shape: {stats[block]['mean'].shape}")
        print(f"Mean range: [{stats[block]['mean'].min()}, {stats[block]['mean'].max()}]")
        print(f"Std shape: {stats[block]['std'].shape}")
        print(f"Std range: [{stats[block]['std'].min()}, {stats[block]['std'].max()}]")

    # Save statistics
    save_path = "./source_feature/souce_features.npy"
    save_statistics(stats, save_path)


if __name__ == '__main__':
    main()