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

