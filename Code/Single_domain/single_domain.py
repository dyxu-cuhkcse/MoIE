#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   single_domain.py
@Time    :   2024/11/11 17:59:07
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   Offline learning class, used as a base class for continual learning
'''

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from lib import utils
from lib.losses import dice_loss
import evaluation
import numpy as np
import logging

class Offline:
    """
    Offline learning class
    - Handles training and evaluation for a single domain.
    """
    def __init__(self, model: nn.Module, args, dir):
        """
        Initialize the offline learning class.
        
        Args:
            model (nn.Module): PyTorch model to train.
            args: Parsed arguments for training configuration.
            dir (str): Directory for saving logs and models.
        """
        super(Offline, self).__init__()
        self.model = model
        self.args = args
        self.bce_criterion = nn.BCELoss()  # Binary Cross-Entropy loss
        self.dice_criterion = dice_loss  # Dice loss for segmentation
        self.log_dir = dir

        # Configure logging
        logging.basicConfig(filename=os.path.join(dir, 'record.log'), level=logging.INFO, force=True)
        logging.info('Initialized offline learning agent for continual learning.')

    def learn_batch(self, site_name, train_loader, test_loaders, writer_dir, background):
        """
        Train the model using batches and periodically evaluate it.
        
        Args:
            site_name (str): Name of the training site/domain.
            train_loader (DataLoader): DataLoader for training data.
            test_loaders (dict): Dictionary of DataLoaders for testing on multiple domains.
            writer_dir (str): Directory for TensorBoard logs.
            background (bool): Whether to include background during training.
        """
        # Configure optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.outer_lr, weight_decay=1e-5)
        writer = SummaryWriter(writer_dir)

        # Initialize training parameters
        train_loader_iter = iter(train_loader)
        itr = 0
        best_val_dice = 0
        best_val_dice_step = 0

        # Main training loop
        while itr < self.args.iterations:
            # Get a batch of data
            imgs, gts, train_loader_iter = utils.batch_from_iterator(train_loader_iter, train_loader)

            # Set the model to training mode
            self.model.train()

            # Forward pass
            outs = self.model(imgs)

            # Compute losses
            bceloss = self.bce_criterion(outs, gts)
            diceloss = self.dice_criterion(outs[:, 1, :, :], gts[:, 1, :, :])
            loss = 0.5 * bceloss + 0.5 * diceloss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            optimizer.step()
            itr += 1

            # Learning rate decay
            if self.args.decay_step is not None and itr % self.args.decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.outer_lr * self.args.decay_rate ** (itr // self.args.decay_step)

            # Logging training progress
            if itr % self.args.print_freq == 0:
                logging.info(f"Iteration = {itr}, lr = {optimizer.param_groups[0]['lr']:.6f}, total_loss = {loss.item():.4f}")

            # Periodic evaluation
            if itr % self.args.test_freq == 0:
                test_dices = {}
                for test_site_name in test_loaders.keys():
                    test_dice = evaluation.evaluation_case(self.model, test_loaders[test_site_name], test_site_name)
                    test_dices[test_site_name] = test_dice

                # Calculate mean test dice across all test sites
                test_dice_mean = np.mean([v for v in test_dices.values()])

                # Log evaluation results
                logging.info(f"-----> MEAN test_dice = {test_dice_mean:.4f}")
                logging.info("-----> Separate test_dice: %s = %.4f, %s = %.4f, %s = %.4f, %s = %.4f, %s = %.4f" %
                             ('BinRushed', test_dices['BinRushed'], 'Drishti_GS', test_dices['Drishti_GS'], 
                              'Magrabia', test_dices['Magrabia'], 'ORIGA', test_dices['ORIGA'], 'REFUGE', test_dices['REFUGE']))

                # Save model checkpoints
                if itr % self.args.save_freq == 0:
                    ckpt_path = os.path.join(
                        utils.check_folder(os.path.join(self.log_dir, 'model')),
                        f"step_{itr}_dice_{test_dice_mean:.4f}.pth"
                    )
                    torch.save(self.model.state_dict(), ckpt_path)
                    logging.info(f"Saved model checkpoint to {ckpt_path}")

                # Save the best model
                if test_dice_mean >= best_val_dice:
                    best_ckpt_path = os.path.join(
                        utils.check_folder(os.path.join(self.log_dir, 'model')),
                        'best_model.pth'
                    )
                    torch.save(self.model.state_dict(), best_ckpt_path)
                    logging.info(f"Best test_dice improved from {best_val_dice:.4f} to {test_dice_mean:.4f} at iteration {itr}")
                    best_val_dice = test_dice_mean
                    best_val_dice_step = itr
                else:
                    logging.info(f"Test_dice did not improve from {best_val_dice:.4f} (best at iteration {best_val_dice_step})")