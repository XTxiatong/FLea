
# We adaot the partition method from FedLab-NLP/fedlab/utils/dataset/functional.py
# For details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

import pandas as pd
from pathlib import Path
import math, random
import torch
import torchaudio

from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

from tqdm import tqdm
from utils.functional import hetero_dir_partition, partition_report,setup_seed,label_skew_quantity_based_partition, client_inner_dirichlet_partition

from operator import itemgetter

## normalised image

def gaussian_noise(x, severity=2):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) 
    return x + np.random.normal(size=x.shape, scale=c)

def contrast(x, severity=2):
    c = [.75, .5, .4, .3, 0.15][severity - 1]

    x = np.array(x)
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return (x - means) * c + means


# others
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, target in dataloader:
        #print(inputs.shape) torch.Size([1, 3, 32, 32])
        #inputs = inputs[:,0,:,:,:]
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean, std)
    return mean, std

def get_target(targets, i, NUM_USERS):
    ## 10 classes: resample  
    indices, targets_sorted = zip(*sorted(enumerate(targets), key=itemgetter(1)))
    #print(targets_sorted)
    #print(indices)
    #print(targets[29],targets[4],targets[6])
    indices = np.array([i for i in indices])
    targets_sorted = np.array([t for t in targets_sorted])
    
    class_starts = [0,5000,10000,15000,20000,25000,30000,35000,40000,45000]
    cnt_class = 5000//(NUM_USERS//100)
    idx = np.linspace(0,len(targets)-1, num=len(targets))
    idx =  [int(i) for i in idx]
    idx_i = []
    for c in range(10):
        IDX = class_starts[c] + cnt_class*i
        idx_i += idx[IDX:IDX+cnt_class]
        
    return targets_sorted[idx_i], indices[idx_i]

def get_target2(targets, i, NUM_USERS):
    ## 6 classes: resample , 
    #base = 75
    CNT = 2
    splits = len(targets)//CNT
    indices = np.array([i for i in range(len(targets))])
    idx_i = indices[i*splits: splits*(i+1)]
    
    print('Pre-splits:', idx_i)
    return targets[idx_i], indices[idx_i]

def split_data(dataset, num_classes, num_clients, split_method, split_para):
    print('Dataset:', dataset)
    if dataset == 'Cifar10':
        mean, std = [0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data
        
        data = trainset.data
        targets = trainset.targets    
        test_targets = testset.targets.tolist()
        test_datas = [testset.data[i].tolist() for i in range(len(test_targets))]
        N_training = 50000
    
    ## download from https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
    ## preprocessing from https://www.kaggle.com/code/longx99/sound-classification
    if dataset == 'UrbanSound':
        metadata_file = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
        data_path = './data/UrbanSound8K/audio/'
        df = pd.read_csv(metadata_file)
        df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
        df = df[['relative_path', 'classID']]
        print(df.head())
        
        myds = SoundDS(df, data_path)

        # Random split of 80:20 between training and validation
        num_items = len(myds)
        num_train = round(num_items * 0.8)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(myds, [num_train, num_val])
        print(len(train_ds), len(val_ds))

        
        data = []
        targets = []
        test_datas = []
        test_targets = []
        for dat, target in tqdm(train_ds):
            data.append(dat) 
            targets.append(target)
        for dat, target in tqdm(val_ds):
            test_targets.append(target) 
            test_datas.append(dat)
            
    ## download from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    ## preprocessing from https://github.com/tengqi159/human-activity-recognition/blob/master/baseline_cnn_uci.py
    if dataset == 'UCI_HAR':
        
        data_path = './data/UCI_HAR/'        
        data = np.load(data_path+'x_train.npy')
        data = data[:,:,:3]
        data = np.reshape(data, [data.shape[0],1, data.shape[1], data.shape[2]])  #add one dimension
        targets = np.load(data_path+'y_train.npy')
        targets = np.argmax(targets, axis=1)
        test_datas = np.load(data_path+'x_test.npy')
        test_datas = test_datas[:,:,:3]
        test_datas = np.reshape(test_datas, [test_datas.shape[0],1,test_datas.shape[1], test_datas.shape[2]])
        test_targets = np.load(data_path+'y_test.npy')
        test_targets = np.argmax(test_targets, axis=1)
      
        print('training:', len(data), 'testing:', len(test_datas))

    ### 
    setup_seed(1)
    NUM_USERS = num_clients    # total number of clients

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': [] ,'local_ood': {},  'global_test':{}}
    #                          local test                          local ood              


    # CREATE USER DATA SPLIT
    if split_method == 'quantity':
        client_dict = label_skew_quantity_based_partition(targets, NUM_USERS, num_classes, major_classes_num = int(split_para))
        major_classes_num = int(split_para)

    elif split_method == 'distribution':
        print('Here')
        if dataset == 'UrbanSound':
            print('Direct:')  
            client_dict = hetero_dir_partition(targets, NUM_USERS, num_classes, dir_alpha=split_para)
        
        elif dataset == 'Cifar10':
            if NUM_USERS <=100:  
                print('Direct:')  
                client_dict = hetero_dir_partition(targets, NUM_USERS, num_classes, dir_alpha=split_para)
            else:
                base = 100
                CNT = NUM_USERS//base
                print(targets)
                client_dict = {}
                for part in range(CNT):
                    target_temp, Idx = get_target(targets, part, NUM_USERS)
                    print(Idx)
                    print(target_temp)
                    if split_para > 0.1:
                        client_dict_temp = hetero_dir_partition(target_temp, base, num_classes, dir_alpha=split_para)
                    else:
                        client_dict_temp = hetero_dir_partition(target_temp, base, num_classes, dir_alpha=split_para,min_require_size=3)
                    for cid in range(base):
                        client_dict[base*part+cid] = [Idx[i] for i in client_dict_temp[cid]]

        elif dataset == 'UCI_HAR':
            if NUM_USERS <=100 or split_para>0.1:  
                print('Direct:')  
                client_dict = hetero_dir_partition(targets, NUM_USERS, num_classes, dir_alpha=split_para)

            else:             
                base = 75
                CNT = NUM_USERS//base
                assert NUM_USERS==150
                client_dict = {}
                for part in range(CNT):
                    target_temp, Idx = get_target2(targets, part, NUM_USERS)
                    print(Idx)
                    print(target_temp)
                    client_dict_temp = hetero_dir_partition(target_temp, base, num_classes, dir_alpha=split_para,min_require_size=3)
                    
                    for cid in range(base):
                        client_dict[base*part+cid] = [Idx[i] for i in client_dict_temp[cid]]

        print('Done')
        major_classes_num = int(num_classes)
        
    partition_report(targets,client_dict,class_num=num_classes,file='data_split_report.txt')


    number = np.zeros([NUM_USERS, num_classes])

    Corrupted_flag = False
 
    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        idx = client_dict[i]

        if dataset == 'Cifar10':
            if Corrupted_flag and i in [0,1,2,3,4,5,6,7,8,9,10]:     
                temp = data[idx]
                temp_data = np.zeros(temp.shape)
                for k in range(temp.shape[0]):
                    temp_data[k] =  temp[k]
                    temp_data[k][:,:,:] = 1 
                X = temp_data.tolist()
                y = targets[idx].tolist()     
                print('mask:', 1)   
            else:   
                X = data[idx].tolist()
                y = targets[idx].tolist()

        if dataset == 'UrbanSound':
            X = [data[k] for k in idx]
            y = [targets[k] for k in idx]
        
        if dataset == 'UCI_HAR':
            X = [data[k] for k in idx]
            y = [targets[k] for k in idx]

        idx = list(range(len(y)))
        random.seed(1)
        random.shuffle(idx)
        idx = idx[:] #use all
  
        for j in idx:
            number[i][y[j]] +=1

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': [X[j] for j in idx], 'y': [y[j] for j in idx]}
        train_data['num_samples'].append(len(idx))

        ###### local testing set, may not be used
        numbers = Counter(y)
        
        present_majority = [l[0] for l in numbers.most_common(major_classes_num)]  #for quanlity distribution it is present class
        present_minority = list(set(range(num_classes))-set(present_majority))   #miniroty class, unpresent
       


        present_idx = []              #local iid
        unpresent_idx = []            #local ood, for dir, may be zero
        for j in range(len(test_targets)):
            if test_targets[j] in present_majority:
                present_idx.append(j) 
            if test_targets[j] in present_minority:
                unpresent_idx.append(j)    
                
        present_X = [test_datas[j] for j in present_idx]
        present_y = [test_targets[j] for j in present_idx]
        test_len = len(present_y)    # local iid testing length
        
         
        random.seed(1)
        random.shuffle(unpresent_idx)
        unpresent_idx = unpresent_idx[:test_len]
        unpresent_X = [test_datas[j] for j in unpresent_idx]
        unpresent_y = [test_targets[j] for j in unpresent_idx]

        if split_method == 'quantity':
            print(i,np.unique(y), np.unique(present_y), np.unique(unpresent_y))
            print(i, 'training:', Counter(y), 'testing iid:', Counter(present_y))#, 'testing ood:', Counter(unpresent_y))
         
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x':  present_X, 'y': present_y}
        test_data['num_samples'].append(test_len)


    test_data['global_test']['x'] = test_datas
    test_data['global_test']['y'] = test_targets ##list variable
 
     
    print("Num_samples of Training set per client:", train_data['num_samples'])
    print("Total_training_samples:", sum(train_data['num_samples']))
    print("Global test set:", len(test_targets))
    print("Finish Generating Samples, distribution saved")

    #np.savetxt('./data/cifar10_' + str(num_clients) + '_'  +str(split_method) + '_' +str(split_para) + '.txt', number,fmt='%d')
    
    return train_data, test_data




class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig])

        return ((resig, sr))
    
    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    
    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
        
# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id
        