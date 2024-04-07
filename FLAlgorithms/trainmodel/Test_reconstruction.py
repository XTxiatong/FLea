import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns

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


def train(model, train_loader, optimizer, epoch, feature_extractor):
    model.train()

    total_loss = []
    Loss = nn.MSELoss()
    for data, target in tqdm(train_loader):
        data = data.cuda()

        feature = feature_extractor.get_feature(data.cuda(), idx=layer)
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
            data = data.cuda()

            feature = feature_extractor.get_feature(data.cuda(), idx=layer)
            prediction = model(feature)
            loss = Loss(data, prediction)

           
            total_loss.append(loss.item())
            
    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Testing Set: Average Loss: {avg_loss:.2f}")

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.05)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()
    
    # ## visualise feature
    # for i in range(10):
        # data, label = test_dataset[i]
        # ax = plt.subplot(1,10, i+1)
        # plt.imshow(transforms.ToPILImage()(data))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    # plt.savefig('example.png')
    
    feature_extractor = MobileNetV2(10).cuda()
    model_path = os.path.join("../../models", 'FedNF')
    checkpoint_path = os.path.join('/auto/homes/tx229/federated/FL_v4/models/Qua_3', "server_FedFea_MOBNET_Cifar10_loss_CE_CE_KL_epoch_10_100.pt")
    print(checkpoint_path)
    feature_extractor = torch.load(checkpoint_path).cuda()
    print('Load model checkpoint from name succuessfully!') 
    print(feature_extractor)
    

    
    model = MinimalDecoder(input_nc=16, output_nc=3, input_dim=32, output_dim=32) #0-16, 1=24
    print(model)
    model = model.cuda()

    kwargs = {"num_workers": 2, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=3000, shuffle=False, **kwargs
    )


    milestones = [25, 50, 80]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, feature_extractor)
        test(model, test_loader, feature_extractor)
        
        scheduler.step()

    torch.save(model.state_dict(), "cifar_reconstruction_model.pt")

    # ## visualise feature
    # for data, target in test_loader:
        # data = data.cuda()
    # feature = feature_extractor.get_feature(data.cuda(), idx=0)
    # prediction = model(feature)
    
    # data = data[:10,:,:,:].cpu().detach()
    # prediction = prediction[:10,:,:,:].cpu().detach()
    # print(data.shape)
        
    # plt.figure(figsize=(20, 4))    
    # for i in range(10):
        # ax = plt.subplot(2,10, i+1)
        # plt.imshow(transforms.ToPILImage()(data[i]))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        
        # ax = plt.subplot(2,10, (i+11))
        # plt.imshow(transforms.ToPILImage()(prediction[i]))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    # plt.savefig('example.png')


   
        
    ## visualise feature by KNN
    for data, target in test_loader:
        data = data.cuda()
    data = data[100:]    
    target = target[100:]    
    
    for c in range(10):
        if c == 0:
            idx = (target == c).nonzero(as_tuple=True)[0][1].view(1)
        else:
            idx = torch.cat((idx,(target == c).nonzero(as_tuple=True)[0][1].view(1)))
    
    data_10 = data[idx]
    target_10 = target[idx]
    
    Loss = nn.MSELoss()
    
    data = torch.cat((data_10,data))
    target = torch.cat((target_10, target))
    feature = feature_extractor.get_feature(data.cuda(), idx=layer)
    feature_10 = feature[:10,]
    
    print('raw data correlation:', dist_corr(data_10.reshape(10,-1).cpu(),feature_10.reshape(10,-1).cpu() ))
    
    print('raw data 10 MSE:', Loss(data_10,data_10).item())
    
    prediction = model(feature)
    print('raw feature MSE:', Loss(data,prediction).item()*3*32*32)
    data = data.cpu().detach()
    prediction = prediction.cpu().detach()
   
    plt.figure(figsize=(20, 10)) 
    plt.title('xx')
    for i in range(10):
        ax = plt.subplot(4,10, i+1)
        plt.imshow(transforms.ToPILImage()(data[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        ax = plt.subplot(4,10, i+11)
        plt.imshow(transforms.ToPILImage()(prediction[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    ## client aggregation    
    distance = torch.cdist(feature.reshape(feature.shape[0],-1), feature.reshape(feature.shape[0],-1), p=2)
    K = 2
   
    nns = []
    for i in range(10):
        _, indices = torch.sort(distance[i,], descending=False)
       
        for j in indices[2:]: #from not itself
            if target[j] == target[i]: 
                break  #alway the final?
        index = [i,j]
        nns.append(j)        
        print(index, target[i], target[j])
        temp = torch.mean(feature[index,], dim=0).reshape(1, feature.shape[1], feature.shape[2], feature.shape[3]) 
        feature[i,] = temp #alreay aggraged

    feature_10 = feature[:10,]
    print('client KNN correlation:', dist_corr(data_10.reshape(10,-1).cpu(),feature_10.reshape(10,-1).cpu() ))
    
    prediction = model(feature)
    
    print('client data MSE:', Loss(data_10,prediction[:10,]).item()*3*32*32)
    
    prediction = prediction.cpu().detach()
    
    
   
    for i in range(10):
        ax = plt.subplot(4,10, (i+21))
        plt.imshow(transforms.ToPILImage()(prediction[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

   
    ## another client
    flag = False
    for data2, target2 in test_loader:
        data2 = data2.cuda()
        flag = True
        if flag:
            break
        
        
    feature2 = feature_extractor.get_feature(data2.cuda(), idx=layer)
    K = 2 
    local_class = torch.unique(target2).cpu().detach().tolist()
    for idx, c in enumerate(local_class):
        feature_c = feature2[target2==c] 
        lab_c = target2[target2==c]
        feature_c_new = torch.zeros(feature_c.shape).to(feature_c.device)
        distance = torch.cdist(feature_c.reshape(feature_c.shape[0],-1), feature_c.reshape(feature_c.shape[0],-1), p=2)
        
        #print(distance)
        for i in range(len(feature_c)):
            _, indices = torch.sort(distance[i,], descending=False)
            index = indices[:K]
            temp = torch.mean(feature_c[index,], dim=0).reshape(1, feature_c.shape[1], feature_c.shape[2], feature_c.shape[3]) 
            #print(temp.shape)
            feature_c_new[i,] = temp
        
        if idx == 0:
            features_new = feature_c_new
            labels_new = lab_c
        else:
            features_new = torch.cat((features_new, feature_c_new), dim=0) 
            labels_new = torch.cat((labels_new, lab_c), dim=0) 
            
    distance2 = torch.cdist(feature.reshape(feature.shape[0],-1), features_new.reshape(features_new.shape[0],-1), p=2)
    for i in range(10):
        _, indices = torch.sort(distance2[i,], descending=False)
       
        for j in indices:
            if target2[j] == target[i]: 
                break
        index = [i,j]
        #nns.append(j)        
        print(index, target[i], target2[j])
        temp = (feature[i,] + features_new[j,])/2
        feature[i,] = temp #alreay aggraged    
        
    prediction = model(feature)
    print('server data MSE:', Loss(data_10,prediction[:10,]).item()*3*32*32)
    prediction = prediction.cpu().detach()    
    for i in range(10):
        ax = plt.subplot(4,10, (i+31))
        plt.imshow(transforms.ToPILImage()(prediction[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
       
    feature_10 = feature[:10,]
    print('server correlation:', dist_corr(data_10.reshape(10,-1).cpu(),feature[-10:,].reshape(10,-1).cpu() ))            
        
        

    plt.savefig('example.png')






if __name__ == "__main__":
    main()