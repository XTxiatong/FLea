import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from FLAlgorithms.trainmodel.models import VGG16, MobileNetV2, VisionTransformer, ResNet18, MinimalDecoder
import os
 
import warnings
warnings.filterwarnings('ignore')

def pairwise_dist(A):
    # Taken frmo https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    #A = torch_print(A, [torch.reduce_sum(A)], message="A is")
    r = torch.sum(A*A, 1)
    r = torch.reshape(r, [-1, 1])
    print(A.shape, r.shape)
    rr = r.repeat(1,A.shape[0])
    rt = r.T.repeat(A.shape[0],1)
    D = torch.maximum(rr - 2*torch.matmul(A, A.T) + rt, 1e-7*torch.ones(A.shape[0], A.shape[0]))
    D = torch.sqrt(D)
    return D

def dist_corr(X, Y):
    n = X.shape[0]
    a = pairwise_dist(X)
    b = pairwise_dist(Y)

    A = a - torch.mean(a,1).repeat(a.shape[1],1).T - torch.mean(a,0).repeat(a.shape[0],1) + torch.mean(a)
    B = b - torch.mean(b,1).repeat(b.shape[1],1).T - torch.mean(b,0).repeat(b.shape[0],1) + torch.mean(b)
    dCovXY = torch.sqrt(torch.sum(A*B) / (n ** 2))
    dVarXX = torch.sqrt(torch.sum(A*A) / (n ** 2))
    dVarYY = torch.sqrt(torch.sum(B*B) / (n ** 2))
    
    dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)
    return dCorXY
 
layer = 0

def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    mean, std = [0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768]
    normalize = transforms.Normalize((mean), (std))
    
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train (default: 50)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.05)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()
    

    feature_extractor = MobileNetV2(10).cuda()
    checkpoint_path = os.path.join('/auto/homes/tx229/federated/FL_v4/models/Qua_3', "server_FedFea_MOBNET_Cifar10_loss_CE_CE_KL_epoch_10_100.pt")
    print(checkpoint_path)
    feature_extractor = torch.load(checkpoint_path).cuda()
    print('Load model checkpoint from name succuessfully!') 
    print(feature_extractor)
    
    kwargs = {"num_workers": 2, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5000, shuffle=True, **kwargs)
    
    layer = 0
    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            feature = feature_extractor.get_feature(data.cuda(), idx=layer)
    
    print(torch.unique(target))
    print(feature.shape)
    cdis = torch.cdist(feature.reshape(feature.shape[0],-1).cpu(), feature.reshape(feature.shape[0],-1).cpu(), p=2)
    print(cdis.shape)
    np.save('dist_l2_5000.npy',cdis.detach() ) 
    
    ### use two KNN
    client1_fea = feature[:1250,]  
    client1_lab = target[:1250,]
    client2_fea = feature[1250:2500,]
    client2_lab = target[1250:2500,]
    print('clients:', torch.unique(client2_lab), torch.unique(client1_lab))
    
    ## client1 aggregation    
    distance = torch.cdist(client1_fea.reshape(client1_fea.shape[0],-1), client1_fea.reshape(client1_fea.shape[0],-1), p=2)
    nns = []
    client1_feanew = torch.zeros(client1_fea.shape)
    for i in range(client1_fea.shape[0]):
        _, indices = torch.sort(distance[i,], descending=False)
        for j in indices[1:]: #from not itself
            if client1_lab[j] == client1_lab[i]: 
                break  #alway the final?
        index = [i,j]
        nns.append(j) 
        temp = torch.mean(client1_fea[index,], dim=0).reshape(1, client1_fea.shape[1], client1_fea.shape[2], client1_fea.shape[3]) 
        client1_feanew[i,] = temp #alreay aggraged
    
    ## client2 aggregation    
    distance = torch.cdist(client2_fea.reshape(client2_fea.shape[0],-1), client2_fea.reshape(client2_fea.shape[0],-1), p=2)
    nns = []
    client2_feanew = torch.zeros(client2_fea.shape)
    for i in range(client2_fea.shape[0]):
        _, indices = torch.sort(distance[i,], descending=False)
        for j in indices[1:]: #from not itself
            if client2_lab[j] == client2_lab[i]: 
                break  #alway the final?
        index = [i,j]
        nns.append(j) 
        temp = torch.mean(client2_fea[index,], dim=0).reshape(1, client2_fea.shape[1], client2_fea.shape[2], client2_fea.shape[3]) 
        client2_feanew[i,] = temp #alreay aggraged
    
    ##server aggragtion
    distance2 = torch.cdist(client1_feanew.reshape(client1_feanew.shape[0],-1), client2_feanew.reshape(client2_feanew.shape[0],-1), p=2)
    feature_new = torch.zeros(client1_fea.shape)
    for i in range(client1_fea.shape[0]):
        _, indices = torch.sort(distance2[i,], descending=False)
       
        for j in indices:
            if client2_lab[j] == client1_lab[i]: 
                break

        temp = (client1_feanew[i,] + client1_feanew[j,])/2
        feature_new[i,] = temp #alreay aggraged    
    
    print(feature_new.shape)
    cdis = torch.cdist(feature.reshape(feature.shape[0],-1).cpu(), feature_new.reshape(feature_new.shape[0],-1).cpu(), p=2)
    print(cdis.shape)
    np.save('dist_l2_KNN.npy',cdis.detach() ) 
    
if __name__ == "__main__":
    main()