import torch
from torch.nn import Module
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import numpy as np
import copy
from torch.autograd import Variable
from math import *
from FLAlgorithms.trainmodel.models import *
from sklearn import metrics
import json
from collections import Counter



# def get_entropy(pros):
    # print(pros.shape)
    # en = np.zeros(len(pros))
    # num_cla = pros.shape[1]
    # for j in range(len(pros)):
        # en[j] = max(pros[j,:])
    # return en

def get_entropy(pros):
    en = np.zeros(len(pros))
    num_cla = pros.shape[1]
    for j in range(len(pros)):
        for i in range(num_cla):
            p = pros[j,i]
            if p == 1:
                p = 0.99999
            if p == 0:
                p = 0.00001
            en[j] += -p*log(p)
    return en
        
def z_score_all(x1,x2):
    x = x1.tolist() + x2.tolist()
    xmin = min(x)
    xmax = max(x)
    return [(s-xmin)/(xmax-xmin) for s in x]        

class User:
    """
    Base class for users in federated learning.
    """

    def __init__(self, numeric_id, train_data, test_data, test_ood, test_gb_ood, model, layer=11, percent=0.1, batch_size=0, learning_rate=0,
                 local_epochs=0, device=torch.device('cpu'), output_dim=10,loss='CE',beta=0.1):
        
        self.output_dim = output_dim
        self.modelname = model[1]
        self.id = int(numeric_id[2:])  # 'f_{0:05d}'.format(i)
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.test_batch_size = 100
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.testloader = DataLoader(test_data, self.test_batch_size) if self.id==0 else []  #leave one user for testing
        self.testallloader = DataLoader(test_gb_ood, self.test_batch_size, shuffle=True) if self.id==0 else [] # avaible
        self.iter_trainloader = iter(self.trainloader)
        self.loss = loss
        self.training_data = train_data
        self.layer = layer
        self.percent = percent

        self.shared = False   #Flag for data sharing
        self.personal_model = copy.deepcopy(model[0])
        self.global_model = copy.deepcopy(model[0])
        self.device = device
        if self.loss == 'Max_Min':
            self.discriminator = MLP(1280,512) ##MobileNet
            self.discriminator.to(device)
        

        self.N_Batch = len(train_data) // batch_size + 1
        self.data_size = len(train_data)
        self.beta = beta
        self.alpha_1 = 1
        self.alpha_2 = 1
        self.alpha_3 = 1
        self.alpha_4 = 1
        
        self.class_fre =  self.get_class_distribution()##train_data disribution, cuda tensor 
        
        self.hold_data = self.send_data(fre=1,fraction=percent)

    def set_parameters(self, model, loss): 
        for key in self.personal_model.state_dict().keys(): 
            self.personal_model.state_dict()[key].data.copy_(model.state_dict()[key])
            self.global_model.state_dict()[key].data.copy_(model.state_dict()[key])
            
    def get_parameters(self):
        return self.personal_model.state_dict()

    def send_features(self):
        self.personal_model.eval()
        with torch.no_grad():  
            if self.layer >= 0:
                features_selected, batch_Y_selected = self.hold_data
                features_selected = self.personal_model.get_feature(features_selected, idx=self.layer)
            else:  #share raw data with negative layers
                features_selected, batch_Y_selected = self.hold_data
        
        return features_selected, batch_Y_selected
    
    def send_data(self,fre,fraction=1.0):
        with torch.no_grad():  
            for batch in range(self.N_Batch):     
                batch_X, batch_Y = self.get_next_train_batch() 
                if batch ==0:
                    Y =  batch_Y #F.one_hot(batch_Y,10)
                    X = batch_X
                else:
                    Y = torch.cat((Y, batch_Y), dim=0) # torch.cat((Y, F.one_hot(batch_Y,10)), dim=0) 
                    X = torch.cat((X, batch_X), dim=0)

        if fre == 1 and fraction < 1.0:
            
            cnt = int(fraction*(len(X)))
            idx = np.random.choice(range(len(X)), cnt).tolist()
            X_selected = X[idx,:,:,:]
            Y_selected = Y[idx]
            #print('share data:', X_selected.shape, Y_selected.shape )  
        
        if fre > 1:  #FedMix, FedBR
        
            cnt = X.shape[0]//fre
            Y = F.one_hot(Y,self.output_dim)
            Y = Y.type(torch.float32)
            X_selected = torch.zeros([cnt,X.shape[1], X.shape[2], X.shape[3]]).to(self.device)
            Y_selected = torch.zeros([cnt,Y.shape[1]]).type(torch.float32).to(self.device)
            for i in range(cnt):
                X_selected[i,:,:,:] = torch.mean(X[i*fre:(i+1)*fre,:,:,:], dim=0)
                Y_selected[i,:] = torch.mean(Y[i*fre:(i+1)*fre,:], dim=0)
            
        del X, Y
        #self.shared = True
        return X_selected, Y_selected
    
    def receive_features(self, features): ## update feature loader
        self.feature_batch = len(features)//self.batch_size + 1
        self.featureloader = DataLoader(features, self.batch_size, shuffle=True)
        self.iter_featureloader = iter(self.featureloader)
        del features
        torch.cuda.empty_cache()
        
    def receive_data(self, features): ##update training loader

        N = len(self.training_data + features)
        print('new training data:', N) 
        self.N_Batch = N//self.batch_size + 1
        self.trainloader = DataLoader(self.training_data +features, self.batch_size, shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        del features
        torch.cuda.empty_cache()

    def extract_features(self, batch_X):
        with torch.no_grad():  
            if self.layer >= 0:
                feature = self.personal_model.get_feature(batch_X, idx=self.layer) 
            else: 
                feature = batch_X
            return feature
   
    def send_feadim(self):
        fea, lab = self.send_features()
        return fea.shape
    
    def get_class_distribution(self):

        class_fre = torch.zeros([self.output_dim])
        for batch in range(self.N_Batch):
            batch_X, batch_Y = self.get_next_train_batch()
            
            if batch == 0:
                all_Y = batch_Y
            else:
                all_Y = torch.cat((all_Y, batch_Y))
                
        hist = torch.bincount(all_Y)
        class_fre[:len(hist)] = hist  #keep dim
        del batch_X, batch_Y
        return class_fre
        
    def get_next_train_batch(self):
        try:
            X = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            X = next(self.iter_trainloader)
        return X
        
    def get_next_feature_batch(self):
        try:
            X = next(self.iter_featureloader)
        except StopIteration:
            self.iter_featureloader = iter(self.featureloader)
            X = next(self.iter_featureloader)
        return X 
        
    def test(self,savename, flag=False):  
        self.personal_model.eval() 
        pd_acc = 0
        num = 0

        for X,Y in self.testloader: 
            data = Variable(X)
            output = self.personal_model(data)
            pd_acc += torch.sum(torch.argmax(output, dim=1) == Y).item()      
            num += X.shape[0]

        print('classification local acc:', pd_acc/num)



        gd_acc = 0
        num_all = 0
 
        save_feature = flag
        if not save_feature:
            for X,Y in self.testallloader: 
                data = Variable(X)
                outputs = self.personal_model(data)
                gd_acc += torch.sum(torch.argmax(outputs, dim=1) == Y).item() 
                num_all += X.shape[0]
        
        else:
            for X,Y in self.testallloader: 
                data = Variable(X)
                
                feature= []
                def hook(module, input, output):
                    feature.append(output)

                if self.personal_modelname == 'VGG':
                    handle1 = self.personal_model.vgg[4].register_forward_hook(hook)
                    handle2 = self.personal_model.vgg[9].register_forward_hook(hook)
                    handle3 = self.personal_model.vgg[16].register_forward_hook(hook)
                    handle4 = self.personal_model.vgg[23].register_forward_hook(hook)
                    handle5 = self.personal_model.vgg[30].register_forward_hook(hook)
                    handle6 = self.personal_model.classifier[7].register_forward_hook(hook)
                
                if self.personal_modelname == 'RESNET': #ResNET18
                    handle1 = self.personal_model.resnet.maxpool.register_forward_hook(hook)
                    handle2 = self.personal_model.resnet.layer1.register_forward_hook(hook)
                    handle3 = self.personal_model.resnet.layer2.register_forward_hook(hook)
                    handle4 = self.personal_model.resnet.layer3.register_forward_hook(hook)
                    handle5 = self.personal_model.resnet.layer4.register_forward_hook(hook)
                    handle6 = self.personal_model.resnet.fc.register_forward_hook(hook)  
                    
                if self.personal_modelname == 'MOBNET': #MOBINET
                    handle1 = self.personal_model.layers[1].register_forward_hook(hook)
                    handle2 = self.personal_model.layers[4].register_forward_hook(hook)
                    handle3 = self.personal_model.layers[8].register_forward_hook(hook)
                    handle4 = self.personal_model.layers[12].register_forward_hook(hook)
                    handle5 = self.personal_model.layers[16].register_forward_hook(hook)
                    handle6 = self.personal_model.bn2.register_forward_hook(hook)
               
               
                outputs = self.personal_model(data)
                
                
                gd_acc += torch.sum(torch.argmax(outputs, dim=1) == Y).item() 
                prediction = torch.argmax(outputs, dim=1) 
                
                for i in range(6):
                    if i==5 and self.personal_modelname == 'MOBNET':
                        out = F.avg_pool2d(F.relu(feature[i]), 4) 
                        feature[i] = out.view(output.size(0), -1)
                    else:
                        feature[i] = feature[i].view(output.size(0), -1)
                
                if num_all == 0:
                    features = [feature[i].cpu().detach().numpy() for i in range(6)]
                    labels = Y.cpu().detach().numpy()
                    predictions = prediction.cpu().detach().numpy()
                else:
                    print(features[0].shape)
                    print(features[1].shape)
                    print(features[2].shape)
                    print(features[3].shape)
                    print(features[4].shape)
                    print(features[5].shape)
                    for i in range(6):
                        features[i] = np.concatenate((features[i],feature[i].cpu().detach().numpy()), axis=0)
                    labels = np.concatenate((labels,Y.cpu().detach().numpy()), axis=0)
                    predictions = np.concatenate((predictions,prediction.cpu().detach().numpy()), axis=0)
                   
                num_all += X.shape[0]
                del feature
                handle1.remove()
                handle2.remove()
                handle3.remove()
                handle4.remove()
                handle5.remove()
                handle6.remove()       
            
            for f in range(6):
                np.savetxt(savename + '_' +str(f)+'_features.txt', features[f], fmt = '%.2f')
            np.savetxt('labels.txt', labels, fmt = '%d')
            np.savetxt(savename + '_' + 'predictions.txt', predictions, fmt = '%d')
            
             
                
        print('classification global acc:', gd_acc/num_all)
            
        
        return pd_acc/num, gd_acc/num_all


