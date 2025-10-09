#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   regular_sequence.py
@Time    :   2025/10/09 13:40:56
@Author  :   Dunyuan XU
@Version :   0.1
@Desc    :   None
'''

import os, random, torch, logging, argparse
import numpy as np
import Unet
import DMoE_trainer
import MoE_unet
import torch
import torch.optim as optim
from augmentation_dataloader import offline_loader, joint_loader, BigAug_loader
from monai.networks.utils import one_hot
from collections import Counter
import numpy as np
import os
import torch.nn as nn
from datetime import datetime


def calculate_dice(pred, gt):
    pred = torch.argmax(pred, dim=1)  # [1, 256, 256]
    gt = torch.argmax(gt, dim=1)      # [1, 256, 256]
    
    dices = []
    for class_id in range(2):
        pred_class = (pred == class_id)
        gt_class = (gt == class_id)
        
        intersection = torch.sum(pred_class * gt_class)
        union = torch.sum(pred_class) + torch.sum(gt_class)
        
        if union == 0:
            dice = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
        else:
            dice = (2.0 * intersection) / union
        
        dices.append(dice.item())
    
    mean_dice = sum(dices) / len(dices)
    return dices, mean_dice

def convert_original_unet_to_moe(original_state_dict, moe_model):
    new_state_dict = {}
    
    layer_mapping = {
        'conv1': 'conv1',
        'conv2': 'conv2',
        'conv3': 'conv3',
        'conv4': 'conv4',
        'center': 'center',
        'up4': 'up4',
        'up3': 'up3',
        'up2': 'up2',
        'up1': 'up1',
        'final': 'final'
    }
    
    for key, value in original_state_dict.items():
        for orig_name, new_name in layer_mapping.items():
            if key.startswith(orig_name):
                new_key = key
                new_state_dict[new_key] = value
                break
    
    moe_model_dict = moe_model.state_dict()
    moe_model_dict.update(new_state_dict)
    moe_model.load_state_dict(moe_model_dict, strict=False)
    
    return moe_model

def setup_optimizer(params, method='Adam', lr=1e-3, beta=0.99, wd=0.0, momentum=None, Damp=None, Nesterov=None):
    if method == 'Adam':
        return optim.Adam(params,
                    lr=lr,
                    betas=(beta, 0.999),
                    weight_decay=wd)
    elif method == 'SGD':
        return optim.SGD(params,
                   lr=lr,
                   momentum=momentum,
                   dampening=Damp,
                   weight_decay=wd,
                   nesterov=Nesterov)
    else:
        raise NotImplementedError

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad for all parameters
    model.requires_grad_(False)
    
    # enable grad only for MoE adapter parts
    for name, module in model.named_modules():
        if isinstance(module, MoE_unet.DynamicMOEAdapter):
            module.requires_grad_(True)
            for expert in module.adapter_experts:
                expert.requires_grad_(True)
        if hasattr(module, 'gate_weight'):
            module.gate_weight.requires_grad_(True)
    return model

def collect_params(model):
    """Collect parameters from MoE adapters.

    Walk the model's modules and collect all MoE adapter parameters.
    Return the parameters and their names.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, MoE_unet.DynamicMOEAdapter):
            if hasattr(m, 'gate_weight'):
                if m.gate_weight.is_leaf:
                    params.append(m.gate_weight)
                    names.append(f"{nm}.gate_weight")
            
            for expert_idx, expert in enumerate(m.adapter_experts):
                for np, p in expert.named_parameters():
                    params.append(p)
                    names.append(f"{nm}.adapter_experts.{expert_idx}.{np}")
    
    return params, names

def load_statistics(load_path):
    loaded_dict = np.load(load_path, allow_pickle=True).item()
    
    stats = {}
    for block in ['conv1', 'conv2', 'conv3', 'conv4']:
        stats[block] = {
            'mean': loaded_dict[f'{block}_mean'],
            'std': loaded_dict[f'{block}_std']
        }
    return stats

def create_DMoE(args, pretrained_model):
    initial_expert_num=args.initial_expert_num
    max_expert_num=args.max_expert_num
    select_num=args.select_num
    warmup_lr = args.warm_up_lr
    adapt_lr = args.adapt_lr
    
    model = MoE_unet.Unet_Dynamic_MoE_everyBlock(initial_expert_num=initial_expert_num, max_expert_num=max_expert_num, select_num=select_num).cuda()
    model_state = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    
    if isinstance(model_state, nn.Module):
        model_state = model_state.state_dict()
    
    moe_model = convert_original_unet_to_moe(model_state, model)
    
    moe_model = configure_model(moe_model)
    params, param_names = collect_params(moe_model)
    optimizer = setup_optimizer(params, lr=adapt_lr)
    episodic = False
    
    MoE_trainer = DMoE_trainer.DMoE(moe_model, pretrained_model, optimizer, warmup_lr=warmup_lr, adapt_lr=adapt_lr, episodic=episodic, max_additional_expert = args.max_warmup_expert)

    return MoE_trainer

def warm_up(args, warm_up_train_loader, MoE_trainer, warm_up_epochs, source_sats):
    print("*"*20+" warm up "+"*"*20)
    logging.info("*"*20+" warm up "+"*"*20)
    MoE_trainer.setup_warmup_lr()
    epoch = 0
    warmup_iter = 0
    while epoch < warm_up_epochs:
        for batch in warm_up_train_loader:
            name, img, gt = batch['slice_name'], batch['img'].cuda(), batch['gt'].cuda()
            output = MoE_trainer(args, img, gt, adaptation=False, source_sats = source_sats, iter=warmup_iter)
            warmup_iter += 1
        epoch += 1
    print("total warm up iter: ", str(warmup_iter))
    logging.info("total warm up iter: "+ str(warmup_iter))
    

def learn_batch(test_loaders, iterations, MoE_trainer, pretrained_model, log_dir, args, sats):
    
    MoE_trainer.setup_adapt_lr()
            
    finetune_dice, previous_dice = [], []
    print("*"*20+" start adaptating "+"*"*20)
    site_index=0
    adapt_iter=0
    for site in test_loaders.keys():
        site_index+=1
        pretrained_model.eval()
        test_loader = test_loaders[site]
        
        dices = []
        pre_dices = []
        for batch in test_loader:
            name, img, gt = batch['slice_name'], batch['img'].cuda(), batch['gt'].cuda()
            output = MoE_trainer(args, img, gt, adaptation=True, source_sats=sats, pretrained_model=pretrained_model, site_index=site_index, iter=adapt_iter)
            assert len(name) == 1 and img.shape[0] == 1 and gt.shape[0] == 1
            pred = one_hot(torch.argmax(output.detach(), dim=1, keepdim=True), num_classes=3, dim=1)
            class_dices, avg_dice = calculate_dice(pred, gt)
            dices.append(avg_dice)
            
        dice_mean = sum(dices) / len(dices)
        finetune_dice.append(dice_mean)
        
    txt_dir = os.path.join(log_dir, 'explaination.txt')
    MoE_trainer.output_results(txt_dir)

    print('-----> Finetune MEAN test_dice = %.4f' % np.mean(finetune_dice))
    print('-----> Separate test_dice: %s = %.4f, %s = %.4f, %s = %.4f, %s = %.4f' 
                % ('Drishti_GS', finetune_dice[0], 'Magrabia', finetune_dice[1], 'ORIGA', finetune_dice[2], 'REFUGE', finetune_dice[3]))

    logging.info('-----> Finetune MEAN test_dice = %.4f' % np.mean(finetune_dice))
    logging.info('-----> Separate test_dice: %s = %.4f, %s = %.4f, %s = %.4f, %s = %.4f' 
                % ('Drishti_GS', finetune_dice[0], 'Magrabia', finetune_dice[1], 'ORIGA', finetune_dice[2], 'REFUGE', finetune_dice[3]))

def parse_args():
    desc = "Pytorch implementation of Meta Learning for Domain Generalization"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=str, default='0', help='train or test or guide')
    parser.add_argument('--ckpt', type=str, default='./exp/debug', help='Directory name to save outputs')
    parser.add_argument('--data_npz', type=str, default='./data/fundus', help='Directory to load data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warm_up_lr', type=float, default=1e-3)
    parser.add_argument('--adapt_lr', type=float, default=1e-3)
    parser.add_argument('--network', type=str, default='Unet', help='Type of training network')
    parser.add_argument('--initial_expert_num', type=int, default=1, help='The number of initial expert')
    parser.add_argument('--max_expert_num', type=int, default=32, help='The number of max expert')
    parser.add_argument('--select_num', type=int, default=32, help='The number of select expert')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='The number of warmup epoch')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the exp result')
    parser.add_argument('--augment', type=int, default=0, help='Whether to augment the data')
    parser.add_argument('--max_warmup_expert', type=int, default=8, help='The number of max expert in warmup stage')
    return parser.parse_args()

def main():
    args = parse_args()
    # -------------------- 实验重复性 ------------------------ #
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.set_default_tensor_type('torch.FloatTensor')

    sites = ['Drishti_GS', 'Magrabia', 'ORIGA', 'REFUGE']
    warm_up_site = ['BinRushed']
    
    if args.network == 'Unet':
        pretrained_model = Unet.Unet_multi_outputs(num_channels=3, num_classes=3).cuda()
    model_state = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    pretrained_model.load_state_dict(model_state)
    
    MoE_trainer = create_DMoE(args, pretrained_model)
    
    log_dir = None
    if args.record:
        print("record result in ./ckpt.")
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        cur_dir = os.path.join("./ckpt/", "order")
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        if args.network == 'Unet':
            log_dir = os.path.join(cur_dir, "order_ours_seed:"+str(seed)+"_network:"+str(args.network)+"_aug:"+str(args.augment)+"-"+ TIMESTAMP)
        else:
            log_dir = os.path.join(cur_dir, "order_ours_seed:"+str(seed)+"_network:"+str(args.network)+"_aug:"+str(args.augment)+"_experts:"+str(args.num_experts)+"-"+ TIMESTAMP)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
    if not args.augment:
        warm_up_train_loader, _, _ = offline_loader(args.data_npz, warm_up_site, args.batch_size)
    else:
        print("load augmentation warm up dataset"+"*"*20)
        warm_up_train_loader, _, _ = BigAug_loader(args.data_npz, warm_up_site, args.batch_size)
        
    _, all_test_loader, _ = offline_loader(args.data_npz, sites, args.batch_size)
    
    source_feature_file = "./source_feature/souce_features.npy"
    sats = load_statistics(source_feature_file)
    
    logging.basicConfig(filename=log_dir+'record.log', level=logging.INFO, force=True)
    logging.info('Explore offline agent for continual learning.')
    warm_up(args, warm_up_train_loader, MoE_trainer, warm_up_epochs=args.warmup_epoch, source_sats=sats)
    
    learn_batch(all_test_loader, 1, MoE_trainer, pretrained_model, log_dir, args, sats)
    

if __name__ == '__main__':
    main()
