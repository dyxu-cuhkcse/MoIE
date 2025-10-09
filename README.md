# MoIE

### Introduction

This repository is for our MICCAI 2025 paper '[Sequence-Independent Continual Test-Time Adaptation with Mixture of Incremental Experts for Cross-Domain Segmentation](https://link.springer.com/chapter/10.1007/978-3-032-05325-1_48)'. 

### Usage

1. Prepare the dataset. Extract the MRI into 2D npz files (slice by slice) and normalize these npz files into [-1, 1]. 

2. Organize the data as following structure:
   ``` 
     ├── data
        ├── Site_A
           ├── train
               ├── sample1.npz, sample2.npz, xxxx
           ├── test
               ├── sample1.npz, sample2.npz, xxxx
        ├── Site_xxxx
        ├── Site_xxxx
   ```
   
3. You need to first train the source model on the source domain by running command:
   ```shell
   cd Single_domain
   python single_domain_train.py --record True --save_freq 200 --background False --network 'Unet'
   ```
   Then extract the feature space by running command:
   ```shell
   python source_std_mean.py --network "Unet" --seed 1 --ckpt "./model/step_3600_dice_0.6668.pth"
   ```
   
4. You can now start running MoIE framwork with pretrained model and source domain feature space obtained in previous step.

   The command for runing under regular sequence setting: 
   ```shell
   cd MoIE
   python regular_sequence.py --network "Unet"  --warmup_epoch 1 --warm_up_lr 1e-3 --adapt_lr 5e-4 --initial_expert_num 2 --max_expert_num 8 --select_num 8 --max_warmup_expert 4 --seed 1 --ckpt "./ckpt/BinRushed/step_3600_dice_0.6668.pth"
   ```
   
   The command for runing under stochastic sequence setting: 
   ```shell
   python stochastice_squence.py --network "Unet" --warmup_epoch 5 --warm_up_lr 1e-3 --adapt_lr 5e-3 --initial_expert_num 2 --max_expert_num 8 --select_num 8 --max_warmup_expert 4 --seed 1 --ckpt "./ckpt/BinRushed/step_3600_dice_0.6668.pth"
   ```

### Citation
If this repository is useful for your research, please consider citing:
```
@inproceedings{xu2025sequence,
  title={Sequence-Independent Continual Test-Time Adaptation with Mixture of Incremental Experts for Cross-Domain Segmentation},
  author={Xu, Dunyuan and Yuan, Yuchen and Zhou, Donghao and Yang, Xikai and Zhang, Jingyang and Li, Jinpeng and Heng, Pheng-Ann},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={502--512},
  year={2025},
  organization={Springer}
}
```

### Questions
Please contact 'dyxu21@cse.cuhk.edu.hk'
