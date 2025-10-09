Data Preparation
    You should organize the data directory as the example I upload in this repo.
    Step1: Extract the MRI into 2D npz files (slice by slice).
    Step2: Normalize these npz files into [-1, 1]. 

Pre-training
    You need to first train the source model on the source domain by running command:
    
    cd Single_domain
    python single_domain_train.py --record True --save_freq 200 --background False --network 'Unet'