#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   single_domain_train.py
@Time    :   2024/11/11 17:59:02
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   Main script for single domain training
'''

import os
import random
import torch
import logging
import argparse
import numpy as np

# -------------------- Ensure Experiment Reproducibility -------------------- #
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy random seed
random.seed(123)  # Python random seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.set_default_tensor_type('torch.FloatTensor')

# -------------------- Import Dependencies -------------------- #
from augmentation_dataloader import offline_loader  # Data loader for augmentations
import Unet  # Import Unet model
from single_domain import Offline  # Offline training class
import torch.nn as nn
from lib.losses import dice_loss  # Custom loss function
from lib import utils
from datetime import datetime

# Initialize loss functions
bce_criterion = nn.BCELoss()  # Binary Cross Entropy loss
dice_criterion = dice_loss  # Dice loss for segmentation tasks

# -------------------- Argument Parsing -------------------- #
def parse_args():
    """
    Parse command-line arguments for configuring training.
    """
    desc = "PyTorch implementation of Meta Learning for Domain Generalization"
    parser = argparse.ArgumentParser(description=desc)

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0', help='GPU device to use')

    # Training operations
    parser.add_argument('--current_site', type=str, default='Vendor_A', help='Name of the dataset site to train on')
    parser.add_argument('--memory_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--data_npz', type=str, default='./data/fundus', help='Path to preprocessed data in npz format')
    parser.add_argument('--data_nii', type=str, default='./data', help='Path to NIfTI data')
    parser.add_argument('--batch_size_presite', type=int, default=8, help='Batch size for the source domain')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--outer_lr', type=float, default=5e-4, help='Learning rate for the outer optimizer')
    parser.add_argument('--clip_norm', type=float, default=10.0, help='Gradient clipping norm')
    parser.add_argument('--decay_step', type=int, default=None, help='Frequency of learning rate decay')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='Learning rate decay factor')

    # Logging, saving, and testing options
    parser.add_argument('--test_freq', type=int, default=200, help='Frequency of testing during training')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts to train')
    parser.add_argument('--save_freq', type=int, default=200, help='Frequency of saving model checkpoints')
    parser.add_argument('--print_freq', type=int, default=50, help='Frequency of logging training progress')
    parser.add_argument('--output_dir', type=str, default='./exp/debug', help='Directory for saving outputs')
    parser.add_argument('--record', type=bool, default=False, help='Flag to enable or disable result recording')
    parser.add_argument('--background', type=bool, default=False, help='Flag to include background during training')
    parser.add_argument('--network', type=str, default='Unet', help='Segmentation network to use (e.g., Unet)')

    # Ensure output directory exists
    utils.check_folder(parser.parse_args().output_dir)

    return parser.parse_args()

# -------------------- Main Function -------------------- #
def main():
    """
    Main training loop for single-domain training.
    """
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Set GPU device

    # Log all training parameters
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")
    
    # Define datasets
    sites = ['BinRushed']  # Site for training
    all_train_loader, _, _ = offline_loader(args.data_npz, sites, args.batch_size_presite)
    
    sites = ['BinRushed', 'Drishti_GS', 'Magrabia', 'ORIGA', 'REFUGE']  # Sites for testing
    _, all_test_loaders, _ = offline_loader(args.data_npz, sites, args.batch_size_presite)

    # Configure the segmentation model
    if args.network == 'Unet':
        model = Unet.Unet(num_channels=3, num_classes=3).cuda()  # Initialize Unet model

    # Configure result recording
    if args.record:
        print("Recording results in ./ckpt.")
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        cur_dir = os.path.join(
            "./ckpt/",
            sites[4]
        )
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        log_dir = os.path.join(cur_dir, "Fundus_Unet:" + "-" + TIMESTAMP)
        writer_dir = os.path.join(log_dir, "summary")
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
    else:
        writer_dir = args.output_dir

    # Initialize the offline trainer
    offline_trainer = Offline(
        model=model, 
        args=args, 
        dir=log_dir
    )

    # Start training
    offline_trainer.learn_batch(
        "BinRushed", 
        train_loader=all_train_loader, 
        test_loaders=all_test_loaders, 
        writer_dir=writer_dir, 
        background=args.background
    )

# -------------------- Entry Point -------------------- #
if __name__ == '__main__':
    main()