# FLea

This repository includes the code to reproduce our experiments in the submission entitled

*FLea: Addressing data scarcity and label skew in federated learning via privacy-preserving feature augmentation*

submitted to KDD 2024  Research Track



## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate FLea
conda env list
```

We mainly use *torch*, *torchvision*, *torchaudio*, *numpy*, *sklearn*,*pandas*



## Code  structure
```
├── FLea
│   ├── data                                           # data path
│   │   ├── CIFAR10
│   │   ├── UrbanSound8K
│   │   ├── UCI_HAR
│   ├── Utils                                          # function path
│   │   ├── data_utils                                 # data partition 
│   │   ├── model_utils
│   │   ├── argparse
│   │   ├── functional
│   ├── FLAlgorithms                          # main algorithm path
│   │   ├── servers                                  # code for server function
│   │   ├── users                                     # code for client function
│   │   ├── trainmodel                           # model architecture
│   ├── run                                           # script for running code
│   │   ├── Image
│   │   ├── Audio
│   │   ├── HAR
│   ├── models                                    # saved models and logs
│   │   ├── saved
│   │   ├── logs
```


## Dataset and split

We use three public datasets, put in `data` path.

- CIFAR10 will be downloaded automatically for the first run. 

- UrbanSound8K can be downloaded from [UrbanSound8K - Urban Sound Datasets (weebly.com)](https://urbansounddataset.weebly.com/urbansound8k.html)

- UCI_HAR can be downloaded from [Human Activity Recognition Using Smartphones - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

  Official training and testing split are used in our experiments. 

**We split the training set data into K clients.**  To simulate label skew, we consider quantity-based skew (Qua(q) with q being the number of presented classes) and distribution-based skew (Dir(u) with u controlling the class skewness).  The partition method is adapted from [FedLab-NLP/fedlab/utils/dataset/functional.py](). To mimic different levels of data scarcity, we distribute the training data to K clients, where K is determined by the average local data size, which is set to as small as 100 and 50 for experiments. 

An example of the CIFAR-10 data distribution is visualized below. The training data for 10 classes are distributed across 100 clients, showcasing a quantity-based skew, with each client having only 3 classes of data. Each column represents one client, and the number of samples is denoted by color.

![cifar10_q3](F:\FL\FLea\cifar10_q3.png)




## Training FLea

There are a few setups in our code:

```
--dataset:  The dataset to use: 'Cifar10', 'UrbanSound', 'UCI_HAR' \
--split_method: The data particiption method: 'quantity','distribution' \
--split_para: Data particiption paraemter: 3, 0.5, 0.1 \
--split_num 1000: The number of clients: K \
--client_num 100: The number of clients per round \
--class_num: The number of classes \
--local_epochs: The number of epoches per round\
--batch_size: Batch size for local training: 32, 64 \
--num_global_iters: The total number of rounds: 100 \
--personal_learning_rate: Local learning rate： 0.001 \
--modelname: Model: 'MOBNET', 'AudioNet','HARNet' \
--algorithm: Our proposed FL method and baselines: 'FLea', 'FedAvg', 'FedProx', 'FedDecor','FedLC', 'FedNTD', 'FedBR', 'CCVR', 'FedMix_pre', 'FedData_pre'\
--loss: Local training lossfuction: 'MCE_DeC_KL', 'CE', 'MCE' \
--layer: From which CNN block the activations are eacted for FLea: 1 \
--fea_percent: The precentage of the data/featuers shared out: 0.1 \
--seed: random seed: 0, 1, 10, 100, 100, 1000 \
```

All the running scripts are provided in folder `./run/`                                           

To run the code for FLea, using CIFAR10 data,  Dir(0.1), K=1000 (average local data size 50)

```
 sh run/Image/run_FLea.sh
```

To run the code for FLea, using audio data, Qua(3), K= 70 (average local data size 100)

```
 sh run/Audio/run_FLea_audio.sh
```

To run the code for FLea, using HAR  data, Dir(0.1), K=150 (average local data size 50)

```
 sh run/HAR/FLea_HAR.sh
```



## Baselines

Baseline implementations are included in our code base. We provide the running script for `FedAvg, FedProx, FedDecor,FedLC, FedNTD, FedBR, CCVR, FedMix, FedData`

To run the code for `FedAvg`, using CIFAR10 data,  distribution skew, K=1000 (average local data size 50)

```
 sh run/Image/run_FedAvg.sh
```

To run the code for FedMix,

```
 sh run/Image/run_FedMix.sh
```



## Results

Our training logs can also be found from `./models/logs/`

For example, the model trained for HAR data, using K=150, partition  Dir(0.1) can be found from `FLea_HAR_K150_Dir0.1.log`


```
================================================================================
Summary of training process:
Dataset                : UCI_HAR
Batch size             : 32
Learing rate           : 0.001
Number of total clients: 150
Split method           : distribution
Split parameter        : 0.1
Clients per round      : 15
Number of global rounds: 100
Number of local rounds : 5
Feature from layer     : 1
Feature percentage     : 0.1
Is interplolated       : False
Local training loss    : MCE_DeC_KL
Loss of beta           : 1.0
Algorithm              : FLea
Modelname              : HARNet
Mode                   : training
Seed                   : 0
================================================================================

Dataset: UCI_HAR
training: 7352 testing: 2947

.....

Number of users per round / total users: 15  /  150
Finished creating FL server.
=== Training starts: algorithm FLea ===
-------------Round number:  0  -------------
......
-------------Round number:  57  -------------

Global Model Acc on global data: 0.6599932134373939 
save a model
......
```


Trained model checkpoints are saved in `./models/saved/`

For example, the model trained for HAR data, using K=150, partition  Dir(0.1) can be found from `./models/saved/server_FLea_HARNet_UCI_HAR_loss_MCE_DeC_KL_epoch_5_100_client_150_split_distribution_0.1.pt`



## Privacy analysis


Training and testing the data reconstruction attack: `python test_test_reconstraction.py`

![recons_example](F:\FL\FLea\recons_example.png)

