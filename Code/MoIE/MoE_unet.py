#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   MoE_unet.py
@Time    :   2025/10/09 13:22:36
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   None
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# -------------------- Simple Adapter -------------------- #
class SimpleAdapter(nn.Module):
    """
    A simple feedforward adapter module with two linear layers and an activation function.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -------------------- Dynamic MoE Adapter -------------------- #
class DynamicMOEAdapter(nn.Module):
    """
    Dynamic Mixture of Experts Adapter for U-Net with support for adding experts during training.
    """
    def __init__(self, dim, initial_expert_num=4, max_expert_num=32, hidden_dim=None, num_k=4):
        """
        Args:
            dim (int): Input dimension.
            initial_expert_num (int): Initial number of experts.
            max_expert_num (int): Maximum number of experts.
            hidden_dim (int): Hidden dimension of the adapter experts.
            num_k (int): Number of active experts used during forward pass.
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.initial_expert_num = initial_expert_num
        self.max_expert_num = max_expert_num
        self.num_k = num_k
        self.new_expert_params = []

        # Initialize experts
        self.current_expert_num = initial_expert_num
        self.adapter_experts = nn.ModuleList([
            SimpleAdapter(in_features=dim, hidden_features=hidden_dim)
            for _ in range(initial_expert_num)
        ])
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer('expert_usage', torch.zeros(max_expert_num))  # Track expert usage

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for the module."""
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def add_expert_init_gating_weight(self, gating_weight):
        """
        Dynamically add a new expert with gating weights.

        Args:
            gating_weight (list): Gating weights for initializing the new expert.

        Returns:
            bool: Whether a new expert was added.
            params: Parameters of the new expert.
            param_names: Names of the new expert's parameters.
        """
        if self.current_expert_num < self.max_expert_num:
            new_expert = SimpleAdapter(in_features=self.dim, hidden_features=self.hidden_dim).cuda()
            assert len(gating_weight) == self.current_expert_num, \
                f"Gating weights length ({len(gating_weight)}) must match current expert number ({self.current_expert_num})"

            weighted_params = {}
            for i, expert in enumerate(self.adapter_experts):
                weight = gating_weight[i]
                for name, param in expert.named_parameters():
                    if name not in weighted_params:
                        weighted_params[name] = torch.zeros_like(param)
                    weighted_params[name] += weight * param.data

            for name, param in new_expert.named_parameters():
                param.data.copy_(weighted_params[name])

            self.adapter_experts.append(new_expert)
            self.new_expert_params.extend(new_expert.parameters())
            self.current_expert_num += 1
            print(f"Adding new expert, current expert number is: {self.current_expert_num}")

            param_names, params = zip(*[(f'expert_{self.current_expert_num}.{name}', param)
                                        for name, param in new_expert.named_parameters()])
            return True, params, param_names
        return False, None, None

    def forward(self, x, assign_expert_index, expert_weight):
        """
        Forward pass for the Dynamic MoE adapter.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            assign_expert_index (int): Index of expert to assign (-1 for weighted sum of experts).
            expert_weight (list): Weights for combining experts.

        Returns:
            torch.Tensor: Output tensor with adapted features.
        """
        B, C, H, W = x.shape
        x_reshape = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # Reshape to [B, H*W, C]

        if assign_expert_index == -1:  # Weighted combination of experts
            if len(expert_weight) == 0:
                return None
            tot_x = torch.zeros_like(x_reshape)
            for idx in range(len(expert_weight)):
                expert_output = self.adapter_experts[idx](x_reshape)
                tot_x += expert_weight[idx] * expert_output
            tot_x = tot_x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return x + tot_x
        else:  # Single expert
            expert_output = self.adapter_experts[assign_expert_index](x_reshape)
            tot_x = expert_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return x + tot_x

# -------------------- U-Net with Dynamic MoE Per Block -------------------- #
class Unet_Dynamic_MoE_everyBlock(nn.Module):
    """
    U-Net architecture with Dynamic MoE adapters integrated at every encoder block.
    """
    def __init__(self, initial_expert_num=4, max_expert_num=8, select_num=4, hidden_dim=[2, 4, 10, 16]):
        """
        Args:
            initial_expert_num (int): Initial number of experts in each adapter.
            max_expert_num (int): Maximum number of experts in each adapter.
            select_num (int): Number of active experts during adaptation.
            hidden_dim (list): Hidden dimensions for adapters at each encoder block.
        """
        super(Unet_Dynamic_MoE_everyBlock, self).__init__()
        self.num_filters = 32
        self.num_channels = 3
        self.num_classes = 3
        filters = [self.num_filters,
                   self.num_filters * 2,
                   self.num_filters * 4,
                   self.num_filters * 8,
                   self.num_filters * 16]

        # Encoder with Dynamic MoE adapters
        self.conv1 = conv_block(self.num_channels, filters[0])
        self.moe1 = DynamicMOEAdapter(filters[0], initial_expert_num, max_expert_num, hidden_dim[0], select_num)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv_block(filters[0], filters[1])
        self.moe2 = DynamicMOEAdapter(filters[1], initial_expert_num, max_expert_num, hidden_dim[1], select_num)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv_block(filters[1], filters[2])
        self.moe3 = DynamicMOEAdapter(filters[2], initial_expert_num, max_expert_num, hidden_dim[2], select_num)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv_block(filters[2], filters[3])
        self.moe4 = DynamicMOEAdapter(filters[3], initial_expert_num, max_expert_num, hidden_dim[3], select_num)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.center = conv_block(filters[3], filters[4], if_dropout=True)

        # Decoder
        self.up4 = UpCatconv(filters[4], filters[3], if_dropout=True)
        self.up3 = UpCatconv(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], self.num_classes, kernel_size=1),
            nn.Softmax2d()
        )

    def forward(self, x, moe1_assign_expert=-1, moe2_assign_expert=-1, moe3_assign_expert=-1, moe4_assign_expert=-1,
                moe1_weights=[], moe2_weights=[], moe3_weights=[], moe4_weights=[]):
        # Encoder with MoE adapters
        conv1 = self.conv1(x)
        moe1 = self.moe1(conv1, moe1_assign_expert, moe1_weights)
        block1_output = moe1 + conv1 if moe1 is not None else conv1

        pool1 = self.pool1(block1_output)
        conv2 = self.conv2(pool1)
        moe2 = self.moe2(conv2, moe2_assign_expert, moe2_weights)
        block2_output = moe2 + conv2 if moe2 is not None else conv2

        pool2 = self.pool2(block2_output)
        conv3 = self.conv3(pool2)
        moe3 = self.moe3(conv3, moe3_assign_expert, moe3_weights)
        block3_output = moe3 + conv3 if moe3 is not None else conv3

        pool3 = self.pool3(block3_output)
        conv4 = self.conv4(pool3)
        moe4 = self.moe4(conv4, moe4_assign_expert, moe4_weights)
        block4_output = moe4 + conv4 if moe4 is not None else conv4

        pool4 = self.pool4(block4_output)
        center = self.center(pool4)

        # Decoder
        up_4 = self.up4(block4_output, center)
        up_3 = self.up3(block3_output, up_4)
        up_2 = self.up2(block2_output, up_3)
        up_1 = self.up1(block1_output, up_2)

        out = self.final(up_1)
        return out, block1_output, block2_output, block3_output, block4_output


# -------------------- Supporting Blocks -------------------- #
class conv_block(nn.Module):
    """
    Convolutional block with two Conv2D layers, BatchNorm, and ReLU.
    """
    def __init__(self, ch_in, ch_out, if_dropout=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.if_dropout = if_dropout
        if self.if_dropout:
            self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.if_dropout:
            return self.dropout(x)
        return x


class UpCatconv(nn.Module):
    """
    Decoder block for U-Net with upsampling and concatenation.
    """
    def __init__(self, in_feat, out_feat, is_deconv=True, if_dropout=False):
        super(UpCatconv, self).__init__()
        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, if_dropout=if_dropout)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, if_dropout=if_dropout)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        outputs = self.up(down_outputs)
        diffY = inputs.size(2) - outputs.size(2)
        diffX = inputs.size(3) - outputs.size(3)
        outputs = F.pad(outputs, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        out = self.conv(torch.cat([inputs, outputs], dim=1))
        return out