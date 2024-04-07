# testing visilization
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torchvision
from torchvision import models, datasets, transforms
from torch import nn
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import seaborn as sns

from FLAlgorithms.trainmodel.models import VGG16, MobileNetV2, VisionTransformer, ResNet18, MinimalDecoder
import os
 
import copy

import warnings
warnings.filterwarnings('ignore')

def pairwise_dist(A):
    # Taken frmo https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    #A = torch_print(A, [torch.reduce_sum(A)], message="A is")
    r = torch.sum(A*A, 1)
    r = torch.reshape(r, [-1, 1])
    rr = r.repeat(1,A.shape[0])
    rt = r.T.repeat(A.shape[0],1)
    D = torch.maximum(rr - 2*torch.matmul(A, A.T) + rt, 1e-7*torch.ones(A.shape[0], A.shape[0]).to(A.device))
    D = torch.sqrt(D)
    return D

def dist_corr(X, F):
    n = X.shape[0]
    a = pairwise_dist(X)
    b = pairwise_dist(F)

    A = a - torch.mean(a,1).repeat(a.shape[1],1).T - torch.mean(a,0).repeat(a.shape[0],1) + torch.mean(a)
    B = b - torch.mean(b,1).repeat(b.shape[1],1).T - torch.mean(b,0).repeat(b.shape[0],1) + torch.mean(b)
    dCovXY = torch.sqrt(torch.sum(A*B) / (n ** 2)+ 1e-7)
    dVarXX = torch.sqrt(torch.sum(A*A) / (n ** 2)+ 1e-7)
    dVarYY = torch.sqrt(torch.sum(B*B) / (n ** 2)+ 1e-7)
    dCorXY = dCovXY / (torch.sqrt(dVarXX + 1e-7) * torch.sqrt(dVarYY+ 1e-7) )
    return dCorXY


class Corelation(nn.Module):
    def __init__(self):
        super(Corelation, self).__init__()
    def forward(self, data, feaure):
        n = data.shape[0]
        loss = dist_corr(data.reshape(n,-1),feaure.reshape(n,-1))
        return loss/n
   
    
class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        
class Normalize(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        super().__init__(mean=mean, std=std)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())        

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
            #normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset


layer = 0
torch.manual_seed(0)
input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()    
mean, std = [0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768]
NI = NormalizeInverse(mean, std)  #inverse normalize
NM = Normalize(mean, std)         #normalizing
 

def train(model, train_loader, optimizer, epoch, feature_extractor):
    model.train()

    total_loss = []
    Loss = nn.MSELoss()
    for data, target in tqdm(train_loader):
    
        data_n = torch.zeros(data.shape)
        for i in range(data.shape[0]):
            data_n[i,:,:,:] = NM(data[i,:,:,:])
        data = data.cuda()

        feature = feature_extractor.get_feature(data_n.cuda(), idx=layer)
        prediction = model(feature)
        loss = Loss(data, prediction)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader, feature_extractor):
    model.eval()
    Loss = nn.MSELoss()
    total_loss = []
    
    for data, target in test_loader:
        with torch.no_grad():
            data_n = torch.zeros(data.shape)
            for i in range(data.shape[0]):
                data_n[i,:,:,:] = NM(data[i,:,:,:])
            data = data.cuda()
            feature = feature_extractor.get_feature(data_n.cuda(), idx=layer)
            prediction = model(feature)
            loss = Loss(data, prediction) 
            total_loss.append(loss.item())
            
    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Testing Set: Average Loss: {avg_loss:.2f}")

    return avg_loss
    

def train_decor(model, train_loader,lossname=None):
  
    
    model.train()
    global_model = copy.deepcopy(model)
    global_model.eval()
   
    tau = 1
    KLLoss = nn.KLDivLoss(reduction='batchmean')
    CELoss = nn.CrossEntropyLoss(reduction='mean')
    CorLoss = Corelation()
    
    Epoch = 30
    losses1 = torch.zeros(Epoch)
    losses2 = torch.zeros(Epoch)
    losses3 = torch.zeros(Epoch)
    
    N_Batch = len(train_loader)
    for epoch in range(Epoch):
        for batch_X, batch_Y in train_loader:
            
            batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
            batch_F =  model.get_feature(batch_X, idx=0)
          
            ## local data CE
            logit = model(batch_X)
            loss1 = CELoss(logit,batch_Y) 
            losses1[epoch] += loss1.item()

            ## local data distilling
            logit_gb = global_model(batch_X)
            pro_gb = F.softmax(logit_gb / tau, dim=1)     ## y
            pro_lc = F.log_softmax(logit / tau, dim=1) ## x 
            loss2 = (tau ** 2) * KLLoss(pro_lc,pro_gb)
            losses2[epoch] += loss2.item()
            
            ##local feature decorrelation
            loss3 = CorLoss(batch_X, batch_F)*20
            losses3[epoch] += loss3.item()
                
            if lossname == 'CE_KL':
                loss_all = loss1 + loss2
            
            elif lossname == 'COR_KL':
                loss_all = loss2 +loss3
                
            elif lossname == 'CE_COR_KL':
                loss_all = loss1 + loss2 +loss3    

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad()        
            loss_all.backward()  #retain_graph=True
            optimizer.step()
            
    print('training:',lossname, 'CEloss:', losses1.mean()/N_Batch, 'KLloss:', losses2.mean()/N_Batch, 'Decorrelation loss:', losses3.mean()/N_Batch)
    print('decorrelation:', losses3/N_Batch)   

    return model

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train (default: 50)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.05)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
   
         
    
    ########save feature
    feature_extractor = MobileNetV2(10).cuda()
    checkpoint_path = os.path.join('FLea/models/saved','c0.4_server_FLea_test_MOBNET_Cifar10_loss_MCE_DeC_KL_epoch_10_500_client_100_split_quantity_3.0.pt')
    feature_extractor = torch.load(checkpoint_path).cuda()
    print('Load model checkpoint from name succuessfully!') 
    feature_extractor.eval()
   
        
    ## reconstruction model (use feature from normalised data)
    model = MinimalDecoder(input_nc=16, output_nc=3, input_dim=32, output_dim=32) #0-16, 1=24
    print(model)
    model = model.cuda()
    
    kwargs = {"num_workers": 2, "pin_memory": True}
    

            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False, **kwargs)

    milestones = [25, 50, 80]
    Loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, epoch, feature_extractor)
        test(model, test_loader, feature_extractor)
        scheduler.step()

    
    
    
    ## visualise reconstruction 
    plt.figure(figsize=(20, 10)) 
    
    data, label = test_dataset[16]
    data[0,2:7,2:7] = 1
    
    ax = plt.subplot(1,2, 1)
    plt.imshow(transforms.ToPILImage()(data))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    data = NM(data)
    data = data[None, :]
    feature = feature_extractor.get_feature(data.cuda(), idx=layer)
    prediction = model(feature)
    Loss = nn.MSELoss()
    print(Loss(data.cuda(), prediction))
    data = data.cpu().detach()
    prediction = prediction.cpu().detach()

    
    ax = plt.subplot(1,2, 2)
    plt.imshow(transforms.ToPILImage()(prediction[0]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    
    plt.savefig('recons_example.png')
    
    
    
if __name__ == "__main__":
    main()