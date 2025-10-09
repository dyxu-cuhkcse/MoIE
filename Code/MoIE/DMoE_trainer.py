#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   DMoE_trainer.py
@Time    :   2025/10/09 13:33:11
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   None
'''

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

class DMoE(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, pretrained_model, optimizer, warmup_lr=1e-3, adapt_lr=1e-3, episodic=False, max_additional_expert=8):
        super().__init__()
        self.model = model
        self.pretrain_model = pretrained_model
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.episodic = episodic
        self.adapt_lr = adapt_lr
        self.bce_criterion = nn.BCELoss()
        self.dice_criterion = dice_loss
        
        self.warmup_conv1_additional_experts = 2
        self.warmup_conv2_additional_experts = 2 
        self.warmup_conv3_additional_experts = 2
        self.warmup_conv4_additional_experts = 2
        
        self.max_additional_expert = max_additional_expert
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        self.domain1_weights = []
        self.domain2_weights = []
        self.domain3_weights = []
        self.domain4_weights = []
        self.domain5_weights = []
        
        self.domain1_counts = []
        self.domain2_counts = []
        self.domain3_counts = []
        self.domain4_counts = []
        self.domain5_counts = []
    
    def setup_warmup_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.warmup_lr

    def setup_adapt_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.adapt_lr
    
    def process_list(self, float_list):
        arr = np.array(float_list)
        
        if arr.max() == arr.min():
            return np.ones_like(arr) / len(arr)
        
        normalized = (arr - arr.min()) / (arr.max() - arr.min())
        exp = np.exp(normalized)
        softmaxed = exp / exp.sum()
        
        return softmaxed.tolist()
    
    def find_max_true_letter_and_value(self, bools, values, letters):
        max_value = float('-inf')
        max_letter = None
        
        for b, v, letter in zip(bools, values, letters):
            if b and v > max_value:
                max_value = v
                max_letter = letter
        return max_letter, max_value if max_value != float('-inf') else (None, None)

    def forward_warmup(self, args, x, gt, sats, max_additional_expert, iter):
        
        blocks = ['conv1', 'conv2', 'conv3', 'conv4']
        pretrained_model_statistics = {}
        pre_output, intermediates = self.pretrain_model(x)
        for block in blocks:
            features = intermediates[block]
            pretrained_model_statistics[block] = features.squeeze().detach().cpu().numpy()
        del pre_output, intermediates, features
        torch.cuda.empty_cache()
        
        block1_similarities, block2_similarities, block3_similarities, block4_similarities = [], [], [], []
        block1_need_reweight, block2_need_reweight, block3_need_reweight, block4_need_reweight = False, False, False, False
        block1_need_new, block2_need_new, block3_need_new, block4_need_new = False, False, False, False
        block1_distance_gap, block2_distance_gap, block3_distance_gap, block4_distance_gap = -1, -1, -1, -1
        adding_index = ['A', 'B', 'C', 'D']
        #deal with block1
        for i in range(self.warmup_conv1_additional_experts):
            _, block1_output, _, _, _ = self.model(x, i, -1, -1, -1, [], [], [], [])
            similarity = self.calculate_single_similarity(sats, block1_output.squeeze().detach().cpu().numpy(), conv_name='conv1')
            block1_similarities.append(similarity)
        gating_weight_1 = self.process_list(block1_similarities)
        _, block1_output, _, _, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, [], [], [])
        distance_for_adding_expert_1 = self.calculate_single_distance(sats, block1_output.squeeze().detach().cpu().numpy(), conv_name='conv1')
        distance_pretrained_model_1 = self.calculate_single_distance(sats, pretrained_model_statistics['conv1'], conv_name='conv1')
        if distance_for_adding_expert_1 > 0.9*distance_pretrained_model_1 and self.warmup_conv1_additional_experts < max_additional_expert:
            block1_need_new = True
            block1_distance_gap =  1*abs(distance_for_adding_expert_1-distance_pretrained_model_1)/distance_pretrained_model_1
            
        for i in range(self.warmup_conv2_additional_experts):
            _, _, block2_output, _, _ = self.model(x, -1, i, -1, -1, gating_weight_1, [], [], [])
            similarity = self.calculate_single_similarity(sats, block2_output.squeeze().detach().cpu().numpy(), conv_name='conv2')
            block2_similarities.append(similarity)
        gating_weight_2 = self.process_list(block2_similarities)
        _, _, block2_output, _, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, [], [])
        distance_for_adding_expert_2 = self.calculate_single_distance(sats, block2_output.squeeze().detach().cpu().numpy(), conv_name='conv2')
        distance_pretrained_model_2 = self.calculate_single_distance(sats, pretrained_model_statistics['conv2'], conv_name='conv2')
        if distance_for_adding_expert_2 > 1*distance_pretrained_model_2 and self.warmup_conv2_additional_experts < max_additional_expert:
            block2_need_new = True
            block2_distance_gap = 1*abs(distance_for_adding_expert_2-distance_pretrained_model_2)/distance_pretrained_model_2
            
        for i in range(self.warmup_conv3_additional_experts):
            _, _, _, block3_output, _ = self.model(x, -1, -1, i, -1, gating_weight_1, gating_weight_2, [], [])
            similarity = self.calculate_single_similarity(sats, block3_output.squeeze().detach().cpu().numpy(), conv_name='conv3')
            block3_similarities.append(similarity)
        gating_weight_3 = self.process_list(block3_similarities)
        _, _, _, block3_output, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, gating_weight_3, [])
        distance_for_adding_expert_3 = self.calculate_single_distance(sats, block3_output.squeeze().detach().cpu().numpy(), conv_name='conv3')
        distance_pretrained_model_3 = self.calculate_single_distance(sats, pretrained_model_statistics['conv3'], conv_name='conv3')
        if distance_for_adding_expert_3 >  1*distance_pretrained_model_3 and self.warmup_conv3_additional_experts < max_additional_expert:
            block3_need_new = True
            block3_distance_gap = 1*abs(distance_for_adding_expert_3-distance_pretrained_model_3)/distance_pretrained_model_3
            
        for i in range(self.warmup_conv4_additional_experts):
            _, _, _, _, block4_output = self.model(x, -1, -1, -1, i, gating_weight_1, gating_weight_2, gating_weight_3, [])
            similarity = self.calculate_single_similarity(sats, block4_output.squeeze().detach().cpu().numpy(), conv_name='conv4')
            block4_similarities.append(similarity)
        gating_weight_4 = self.process_list(block4_similarities)
        _, _, _, _, block4_output = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, gating_weight_3, gating_weight_4)
        distance_for_adding_expert_4 = self.calculate_single_distance(sats, block4_output.squeeze().detach().cpu().numpy(), conv_name='conv4')
        distance_pretrained_model_4 = self.calculate_single_distance(sats, pretrained_model_statistics['conv4'], conv_name='conv4')
        if distance_for_adding_expert_4 >  0.85*distance_pretrained_model_4 and self.warmup_conv4_additional_experts < max_additional_expert:
            block4_need_new = True
            block4_distance_gap = 1*abs(distance_for_adding_expert_4-distance_pretrained_model_4)/distance_pretrained_model_4
            
        letter, value = self.find_max_true_letter_and_value([block1_need_new, block2_need_new, block3_need_new, block4_need_new], [block1_distance_gap, block2_distance_gap, block3_distance_gap, block4_distance_gap], adding_index)
        if letter == "A":
            success, new_params, param_names = self.model.moe1.add_expert_init_gating_weight(gating_weight_1)
            if success:
                block1_need_reweight = True
                self.warmup_conv1_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block1_need_reweight:
                _, block1_output, _, _, _ = self.model(x, self.warmup_conv1_additional_experts-1, -1, -1, -1, [], [], [], [])
                similarity = self.calculate_single_similarity(sats, block1_output.squeeze().detach().cpu().numpy(), conv_name='conv1')
                block1_similarities.append(similarity)
                gating_weight_1 = self.process_list(block1_similarities)
        elif letter == "B":
            success, new_params, param_names = self.model.moe2.add_expert_init_gating_weight(gating_weight_2)
            if success:
                block2_need_reweight = True
                self.warmup_conv2_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block2_need_reweight:
                _, _, block2_output, _, _ = self.model(x, -1, self.warmup_conv2_additional_experts-1, -1, -1, gating_weight_1, [], [], [])
                similarity = self.calculate_single_similarity(sats, block2_output.squeeze().detach().cpu().numpy(), conv_name='conv2')
                block2_similarities.append(similarity)
                gating_weight_2 = self.process_list(block2_similarities)
        elif letter == "C":
            success, new_params, param_names = self.model.moe3.add_expert_init_gating_weight(gating_weight_3)
            if success:
                block3_need_reweight = True
                self.warmup_conv3_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block3_need_reweight:
                _, _, _, block3_output, _ = self.model(x, -1, -1, self.warmup_conv3_additional_experts-1, -1, gating_weight_1, gating_weight_2, [], [])
                similarity = self.calculate_single_similarity(sats, block3_output.squeeze().detach().cpu().numpy(), conv_name='conv3')
                block3_similarities.append(similarity)
                gating_weight_3 = self.process_list(block3_similarities)
        elif letter == "D":
            success, new_params, param_names = self.model.moe4.add_expert_init_gating_weight(gating_weight_4)
            if success:
                block4_need_reweight = True
                self.warmup_conv4_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block4_need_reweight:
                _, _, _, _, block4_output = self.model(x, -1, -1, -1, self.warmup_conv4_additional_experts-1, gating_weight_1, gating_weight_2, gating_weight_3, [])
                similarity = self.calculate_single_similarity(sats, block4_output.squeeze().detach().cpu().numpy(), conv_name='conv4')
                block4_similarities.append(similarity)
                gating_weight_4 = self.process_list(block4_similarities)
                
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        outputs, _, _, _, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, gating_weight_3, gating_weight_4)
        
        bceloss = self.bce_criterion(outputs, gt)
        diceloss = self.dice_criterion(outputs[:,1,:,:], gt[:,1,:,:])
        loss = 0.5 * bceloss + 0.5 * diceloss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def forward(self, args, x, gt, adaptation=True, source_sats=None, pretrained_model=None, site_index=-1, iter=0):
        outputs = None
        if not adaptation:
            outputs = self.forward_warmup(args, x, gt, source_sats, max_additional_expert=self.max_additional_expert, iter=iter)
        else:
            outputs = self.forward_and_adapt(args, x, self.optimizer, source_sats, site_index, iter=iter)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
    def forward_and_adapt(self, args, x, optimizer, sats, site_index, iter):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        blocks = ['conv1', 'conv2', 'conv3', 'conv4']
        pretrained_model_statistics = {}
        pre_output, intermediates = self.pretrain_model(x)
        for block in blocks:
            features = intermediates[block]
            pretrained_model_statistics[block] = features.squeeze().detach().cpu().numpy()
        del pre_output, intermediates, features
        torch.cuda.empty_cache()
        
        block1_similarities, block2_similarities, block3_similarities, block4_similarities = [], [], [], []
        block1_need_reweight, block2_need_reweight, block3_need_reweight, block4_need_reweight = False, False, False, False
        block1_need_new, block2_need_new, block3_need_new, block4_need_new = False, False, False, False
        block1_distance_gap, block2_distance_gap, block3_distance_gap, block4_distance_gap = -1, -1, -1, -1
        adding_index = ['A', 'B', 'C', 'D']
        #deal with block1
        for i in range(self.warmup_conv1_additional_experts):
            _, block1_output, _, _, _ = self.model(x, i, -1, -1, -1, [], [], [], [])
            similarity = self.calculate_single_similarity(sats, block1_output.squeeze().detach().cpu().numpy(), conv_name='conv1')
            block1_similarities.append(similarity)
        gating_weight_1 = self.process_list(block1_similarities)
        _, block1_output, _, _, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, [], [], [])
        distance_for_adding_expert_1 = self.calculate_single_distance(sats, block1_output.squeeze().detach().cpu().numpy(), conv_name='conv1')
        distance_pretrained_model_1 = self.calculate_single_distance(sats, pretrained_model_statistics['conv1'], conv_name='conv1')
        if distance_for_adding_expert_1 > 0.65*distance_pretrained_model_1:
            block1_need_new = True
            block1_distance_gap = 1.1*abs(distance_for_adding_expert_1-distance_pretrained_model_1)/distance_pretrained_model_1
            
        for i in range(self.warmup_conv2_additional_experts):
            _, _, block2_output, _, _ = self.model(x, -1, i, -1, -1, gating_weight_1, [], [], [])
            similarity = self.calculate_single_similarity(sats, block2_output.squeeze().detach().cpu().numpy(), conv_name='conv2')
            block2_similarities.append(similarity)
        gating_weight_2 = self.process_list(block2_similarities)
        _, _, block2_output, _, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, [], [])
        distance_for_adding_expert_2 = self.calculate_single_distance(sats, block2_output.squeeze().detach().cpu().numpy(), conv_name='conv2')
        distance_pretrained_model_2 = self.calculate_single_distance(sats, pretrained_model_statistics['conv2'], conv_name='conv2')
        if distance_for_adding_expert_2 > 0.65*distance_pretrained_model_2:
            block2_need_new = True
            block2_distance_gap = 1.25*abs(distance_for_adding_expert_2-distance_pretrained_model_2)/distance_pretrained_model_2
            
        for i in range(self.warmup_conv3_additional_experts):
            _, _, _, block3_output, _ = self.model(x, -1, -1, i, -1, gating_weight_1, gating_weight_2, [], [])
            similarity = self.calculate_single_similarity(sats, block3_output.squeeze().detach().cpu().numpy(), conv_name='conv3')
            block3_similarities.append(similarity)
        gating_weight_3 = self.process_list(block3_similarities)
        _, _, _, block3_output, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, gating_weight_3, [])
        distance_for_adding_expert_3 = self.calculate_single_distance(sats, block3_output.squeeze().detach().cpu().numpy(), conv_name='conv3')
        distance_pretrained_model_3 =  self.calculate_single_distance(sats, pretrained_model_statistics['conv3'], conv_name='conv3')
        if distance_for_adding_expert_3 > 1*distance_pretrained_model_3:
            block3_need_new = True
            block3_distance_gap = 0.65*abs(distance_for_adding_expert_3-distance_pretrained_model_3)/distance_pretrained_model_3
            
        for i in range(self.warmup_conv4_additional_experts):
            _, _, _, _, block4_output = self.model(x, -1, -1, -1, i, gating_weight_1, gating_weight_2, gating_weight_3, [])
            similarity = self.calculate_single_similarity(sats, block4_output.squeeze().detach().cpu().numpy(), conv_name='conv4')
            block4_similarities.append(similarity)
        gating_weight_4 = self.process_list(block4_similarities)
        _, _, _, _, block4_output = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, gating_weight_3, gating_weight_4)
        distance_for_adding_expert_4 = self.calculate_single_distance(sats, block4_output.squeeze().detach().cpu().numpy(), conv_name='conv4')
        distance_pretrained_model_4 = self.calculate_single_distance(sats, pretrained_model_statistics['conv4'], conv_name='conv4')
        if distance_for_adding_expert_4 >  0.9*distance_pretrained_model_4:
            block4_need_new = True
            block4_distance_gap = 1*abs(distance_for_adding_expert_4-distance_pretrained_model_4)/distance_pretrained_model_4
            
        letter, value = self.find_max_true_letter_and_value([block1_need_new, block2_need_new, block3_need_new, block4_need_new], [block1_distance_gap, block2_distance_gap, block3_distance_gap, block4_distance_gap], adding_index)
        
        if letter == "A":
            success, new_params, param_names = self.model.moe1.add_expert_init_gating_weight(gating_weight_1)
            if success:
                block1_need_reweight = True
                self.warmup_conv1_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block1_need_reweight:
                _, block1_output, _, _, _ = self.model(x, self.warmup_conv1_additional_experts-1, -1, -1, -1, [], [], [], [])
                similarity = self.calculate_single_similarity(sats, block1_output.squeeze().detach().cpu().numpy(), conv_name='conv1')
                block1_similarities.append(similarity)
                gating_weight_1 = self.process_list(block1_similarities)
        elif letter == "B":
            success, new_params, param_names = self.model.moe2.add_expert_init_gating_weight(gating_weight_2)
            if success:
                block2_need_reweight = True
                self.warmup_conv2_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block2_need_reweight:
                _, _, block2_output, _, _ = self.model(x, -1, self.warmup_conv2_additional_experts-1, -1, -1, gating_weight_1, [], [], [])
                similarity = self.calculate_single_similarity(sats, block2_output.squeeze().detach().cpu().numpy(), conv_name='conv2')
                block2_similarities.append(similarity)
                gating_weight_2 = self.process_list(block2_similarities)
        elif letter == "C":
            success, new_params, param_names = self.model.moe3.add_expert_init_gating_weight(gating_weight_3)
            if success:
                block3_need_reweight = True
                self.warmup_conv3_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block3_need_reweight:
                _, _, _, block3_output, _ = self.model(x, -1, -1, self.warmup_conv3_additional_experts-1, -1, gating_weight_1, gating_weight_2, [], [])
                similarity = self.calculate_single_similarity(sats, block3_output.squeeze().detach().cpu().numpy(), conv_name='conv3')
                block3_similarities.append(similarity)
                gating_weight_3 = self.process_list(block3_similarities)
        elif letter == "D":
            success, new_params, param_names = self.model.moe4.add_expert_init_gating_weight(gating_weight_4)
            if success:
                block4_need_reweight = True
                self.warmup_conv4_additional_experts += 1
                self.optimizer.add_param_group({'params': new_params, 'lr': self.optimizer.param_groups[0]['lr']})
            if block4_need_reweight:
                _, _, _, _, block4_output = self.model(x, -1, -1, -1, self.warmup_conv4_additional_experts-1, gating_weight_1, gating_weight_2, gating_weight_3, [])
                similarity = self.calculate_single_similarity(sats, block4_output.squeeze().detach().cpu().numpy(), conv_name='conv4')
                block4_similarities.append(similarity)
                gating_weight_4 = self.process_list(block4_similarities)
                
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        gating_weights = [gating_weight_1, gating_weight_2, gating_weight_3, gating_weight_4]
        if site_index == 1:
            if len(self.domain1_weights) == 0:
                for i in range(len(gating_weights)):
                    self.domain1_weights.append(gating_weights[i])
                    self.domain1_counts.append([1]*len(gating_weights[i]))
            else:
                for i in range(len(gating_weights)):
                    if len(gating_weights[i]) == len(self.domain1_weights[i]):
                        for weight in range(len(gating_weights[i])):
                            self.domain1_weights[i][weight] += gating_weights[i][weight]
                            self.domain1_counts[i][weight] += 1
                    elif len(gating_weights[i]) > len(self.domain1_weights[i]):
                        for weight in range(len(self.domain1_weights[i])):
                            self.domain1_weights[i][weight] += gating_weights[i][weight]
                            self.domain1_counts[i][weight] += 1
                        for weight in range(len(self.domain1_weights[i]), len(gating_weights[i])):
                            self.domain1_weights[i].append(gating_weights[i][weight])
                            self.domain1_counts[i].append(1)
        elif site_index == 2:
            if len(self.domain2_weights) == 0:
                for i in range(len(gating_weights)):
                    self.domain2_weights.append(gating_weights[i])
                    self.domain2_counts.append([1]*len(gating_weights[i]))
            else:
                for i in range(len(gating_weights)):
                    if len(gating_weights[i]) == len(self.domain2_weights[i]):
                        for weight in range(len(gating_weights[i])):
                            self.domain2_weights[i][weight] += gating_weights[i][weight]
                            self.domain2_counts[i][weight] += 1
                    elif len(gating_weights[i]) > len(self.domain2_weights[i]):
                        for weight in range(len(self.domain2_weights[i])):
                            self.domain2_weights[i][weight] += gating_weights[i][weight]
                            self.domain2_counts[i][weight] += 1
                        for weight in range(len(self.domain2_weights[i]), len(gating_weights[i])):
                            self.domain2_weights[i].append(gating_weights[i][weight])
                            self.domain2_counts[i].append(1)
        elif site_index == 3:
            if len(self.domain3_weights) == 0:
                for i in range(len(gating_weights)):
                    self.domain3_weights.append(gating_weights[i])
                    self.domain3_counts.append([1]*len(gating_weights[i]))
            else:
                for i in range(len(gating_weights)):
                    if len(gating_weights[i]) == len(self.domain3_weights[i]):
                        for weight in range(len(gating_weights[i])):
                            self.domain3_weights[i][weight] += gating_weights[i][weight]
                            self.domain3_counts[i][weight] += 1
                    elif len(gating_weights[i]) > len(self.domain3_weights[i]):
                        for weight in range(len(self.domain3_weights[i])):
                            self.domain3_weights[i][weight] += gating_weights[i][weight]
                            self.domain3_counts[i][weight] += 1
                        for weight in range(len(self.domain3_weights[i]), len(gating_weights[i])):
                            self.domain3_weights[i].append(gating_weights[i][weight])
                            self.domain3_counts[i].append(1)
        elif site_index == 4:
            if len(self.domain4_weights) == 0:
                for i in range(len(gating_weights)):
                    self.domain4_weights.append(gating_weights[i])
                    self.domain4_counts.append([1]*len(gating_weights[i]))
            else:
                for i in range(len(gating_weights)):
                    if len(gating_weights[i]) == len(self.domain4_weights[i]):
                        for weight in range(len(gating_weights[i])):
                            self.domain4_weights[i][weight] += gating_weights[i][weight]
                            self.domain4_counts[i][weight] += 1
                    elif len(gating_weights[i]) > len(self.domain4_weights[i]):
                        for weight in range(len(self.domain4_weights[i])):
                            self.domain4_weights[i][weight] += gating_weights[i][weight]
                            self.domain4_counts[i][weight] += 1
                        for weight in range(len(self.domain4_weights[i]), len(gating_weights[i])):
                            self.domain4_weights[i].append(gating_weights[i][weight])
                            self.domain4_counts[i].append(1)
        elif site_index == 5:
            if len(self.domain5_weights) == 0:
                for i in range(len(gating_weights)):
                    self.domain5_weights.append(gating_weights[i])
                    self.domain5_counts.append([1]*len(gating_weights[i]))
            else:
                for i in range(len(gating_weights)):
                    if len(gating_weights[i]) == len(self.domain5_weights[i]):
                        for weight in range(len(gating_weights[i])):
                            self.domain5_weights[i][weight] += gating_weights[i][weight]
                            self.domain5_counts[i][weight] += 1
                    elif len(gating_weights[i]) > len(self.domain5_weights[i]):
                        for weight in range(len(self.domain5_weights[i])):
                            self.domain5_weights[i][weight] += gating_weights[i][weight]
                            self.domain5_counts[i][weight] += 1
                        for weight in range(len(self.domain5_weights[i]), len(gating_weights[i])):
                            self.domain5_weights[i].append(gating_weights[i][weight])
                            self.domain5_counts[i].append(1)
        if block1_need_reweight or block2_need_reweight or block3_need_reweight or block4_need_reweight:
            repeat_time = 1
        else:
            repeat_time = 1
        
        for time in range(repeat_time):
            outputs, _, _, _, _ = self.model(x, -1, -1, -1, -1, gating_weight_1, gating_weight_2, gating_weight_3, gating_weight_4)
            current_statistics = {'conv1': block1_output.squeeze().detach().cpu().numpy(), 'conv2': block2_output.squeeze().detach().cpu().numpy(), 'conv3': block3_output.squeeze().detach().cpu().numpy(), 'conv4': block4_output.squeeze().detach().cpu().numpy()}
            pretrained_similarities_distance, current_similarities_distance, similarity_loss_distance =  self.calculate_distance(sats, pretrained_model_statistics, current_statistics)
            loss = softmax_entropy(outputs).mean(0)
            loss = loss.mean() + similarity_loss_distance
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return outputs

    def output_results(self, txt_dir):
        avg_weight1 = [[self.domain1_weights[i][j]/self.domain1_counts[i][j] for j in range(len(self.domain1_counts[i]))] for i in range(len(self.domain1_counts))]
        avg_weight2 = [[self.domain2_weights[i][j]/self.domain2_counts[i][j] for j in range(len(self.domain2_counts[i]))] for i in range(len(self.domain2_counts))]
        avg_weight3 = [[self.domain3_weights[i][j]/self.domain3_counts[i][j] for j in range(len(self.domain3_counts[i]))] for i in range(len(self.domain3_counts))]
        avg_weight4 = [[self.domain4_weights[i][j]/self.domain4_counts[i][j] for j in range(len(self.domain4_counts[i]))] for i in range(len(self.domain4_counts))]
        avg_weight5 = [[self.domain5_weights[i][j]/self.domain5_counts[i][j] for j in range(len(self.domain5_counts[i]))] for i in range(len(self.domain5_counts))]
        with open(txt_dir, 'w') as f:
            f.write(str(avg_weight1))
            f.write(str(avg_weight2))
            f.write(str(avg_weight3))
            f.write(str(avg_weight4))
            f.write(str(avg_weight5))

    def calculate_distance(self, sats, pretrained_model_statistics, current_statistics, lambda_weight=1.0):
        pretrained_results = {}
        current_results = {}
        similarity_loss = 0
        
        for conv_name in ['conv1', 'conv2', 'conv3', 'conv4']:
            mean1 = sats[conv_name]['mean'].reshape(1, -1)
            std1 = sats[conv_name]['std'].reshape(1, -1)
            if len(pretrained_model_statistics[conv_name]) == 4:
                pretrained_result = pretrained_model_statistics[conv_name].mean(axis=0, keepdims=True).reshape(1, -1)
                current_result = current_statistics[conv_name].mean(axis=0, keepdims=True).reshape(1, -1)
            else:                
                pretrained_result = pretrained_model_statistics[conv_name].reshape(1, -1)
                current_result = current_statistics[conv_name].reshape(1, -1)
            
            pretrained_result_mean = np.mean(pretrained_result, axis=0)
            pretrained_result_std = np.std(pretrained_result, axis=0)
            current_result_mean = np.mean(current_result, axis=0)
            current_result_std = np.std(current_result, axis=0)
            
            pretrained_mean_cos = 1-self.fast_cosine_similarity(mean1, pretrained_result_mean)
            pretrained_std_cos = 1-self.fast_cosine_similarity(std1, pretrained_result_std)
            current_mean_cos = 1-self.fast_cosine_similarity(mean1, current_result_mean)
            current_std_cos = 1-self.fast_cosine_similarity(std1, current_result_std)
            
            pretrained_similarity_distance = lambda_weight * pretrained_std_cos + pretrained_mean_cos
            pretrained_results[f'{conv_name}'] = pretrained_similarity_distance
            current_similarity_distance = lambda_weight * current_std_cos + current_mean_cos
            current_results[f'{conv_name}'] = current_similarity_distance
            
            similarity_loss += abs(current_similarity_distance - pretrained_similarity_distance)
        
        return pretrained_results, current_results, similarity_loss
    
    def calculate_single_similarity(self, sats, current_statistics, conv_name):
        mean1 = sats[conv_name]['mean'].reshape(1, -1)
        std1 = sats[conv_name]['std'].reshape(1, -1)
        
        if len(current_statistics) == 4:
            current_result = current_statistics.mean(axis=0, keepdims=True).reshape(1, -1)
        else:                
            current_result = current_statistics.reshape(1, -1)
        current_result_mean = np.mean(current_result, axis=0)
        current_result_std = np.std(current_result, axis=0)    
        
        current_mean_cos = self.fast_cosine_similarity(mean1, current_result_mean)
        current_std_cos = self.fast_cosine_similarity(std1, current_result_std)
        
        current_similarity_distance = current_std_cos + current_mean_cos
        return current_similarity_distance.item()
    
    def calculate_single_distance(self, sats, current_statistics, conv_name):
        mean1 = sats[conv_name]['mean'].reshape(1, -1)
        std1 = sats[conv_name]['std'].reshape(1, -1)
        
        if len(current_statistics) == 4:
            current_result = current_statistics.mean(axis=0, keepdims=True).reshape(1, -1)
        else:                
            current_result = current_statistics.reshape(1, -1)
        current_result_mean = np.mean(current_result, axis=0)
        current_result_std = np.std(current_result, axis=0)    
        
        current_mean_cos = 1-self.fast_cosine_similarity(mean1, current_result_mean)
        current_std_cos = 1-self.fast_cosine_similarity(std1, current_result_std)
        
        current_similarity_distance = current_std_cos + current_mean_cos
        return current_similarity_distance.item()

    def fast_cosine_similarity(self, mean1, pretrained_result):
        if isinstance(mean1, np.ndarray):
            mean1 = torch.from_numpy(mean1).cuda().float()
            pretrained_result = torch.from_numpy(pretrained_result).cuda().float()
        
        similarity = torch.nn.functional.cosine_similarity(
            mean1, pretrained_result, dim=0
        )
        
        return similarity.mean()
    
    def kl_divergence(self, p, q):
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).cuda().float()
            q = torch.from_numpy(q).cuda().float()
        
        eps = 1e-8
        
        p = torch.nn.functional.softmax(p, dim=-1)
        q = torch.nn.functional.softmax(q, dim=-1)
        
        kl = torch.sum(p * torch.log((p + eps)/(q + eps)))
        
        return kl

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"