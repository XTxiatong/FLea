import torch
import os
import numpy as np
import copy
from torch.nn import Module
import sys
from collections import Counter

class Server:
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate,
                 num_glob_iters, layer, percent, local_epochs, optimizer, num_users, times, device):

        # Set up the main attributes
        self.dataset = dataset
        self.device = device
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.test_batch_size = 128
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model[0]) if isinstance(model[0], Module) else model[0]().to(device)  # global model.
        self.modelpara = {}
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_per_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], [], []
        self.times = times
        self.layer = layer
        self.features = {}
        self.percent = percent

    def send_parameters(self,loss):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.selected_users:
            user.set_parameters(self.model,loss)            
            
    def aggregate_parameters(self, glob_iter=0):
        assert (self.users is not None and len(self.users) > 0)
            
        total_train = 0 
        for user in self.selected_users:
            total_train += user.train_samples
            
            
        all_dicts = {}
        for user in self.selected_users:
            all_dicts[user] = user.get_parameters()
            
        num = 0
        for key in self.model.state_dict().keys(): 
            if 'num_batches_tracked' in key:
                #print('yes') #2 BN
                self.model.state_dict()[key].data.copy_(all_dicts[user][key]) ##las user
            else:
                temp = torch.zeros_like(self.model.state_dict()[key])
                
                for idx, user in enumerate(self.selected_users):
                    temp += user.train_samples/total_train  * all_dicts[user][key]
                    ##add here
                    num += self.model.state_dict()[key].numel()
                self.model.state_dict()[key].data.copy_(temp)

    def copymodel(self):
        self.modelpara = self.model.state_dict() 

    def pastemodel(self):
        for key in self.model.state_dict().keys(): 
            self.model.state_dict()[key].data.copy_(self.modelpara[key])
                    
    def send_features(self):  # all users or selected users
        assert (self.users is not None and len(self.users) > 0)
        for i, user in enumerate(self.selected_users):
            if i == 0:
                features, labels = user.send_features()   
            else:
                feature, label = user.send_features() 
                features = torch.cat((features, feature), dim=0)
                labels = torch.cat((labels, label))
        
        privacy_aggragation = self.is_interpolated
        
        if not privacy_aggragation:
            all_features = [(x, y) for x, y in zip(features, labels)] #should shuffle
            print('all data:', len(all_features))
        
        else: ##KNN and down sample, half
            # add privacy protection by nearest neighboor aggragtion
            K = 2
            precent = 1.0/K
            local_class = torch.unique(labels).cpu().detach().tolist()
            #pdist = nn.PairwiseDistance(p=2)
            for idx, c in enumerate(local_class):
                feature_c = features[labels==c] 
                lab_c = labels[labels==c]
                feature_c_new = torch.zeros(feature_c.shape).to(feature_c.device)
                #distance = pdist(feature_c.reshape(feature_c.shape[0],-1), feature_c.reshape(feature_c.shape[0],-1))  #distance needs reshape
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
            
            print( 'K', K, features_new.shape, labels_new.shape) 
            
            cnt = int(precent*(len(features)))
            idx = np.random.choice(range(len(features)), cnt).tolist()
            features_selected = features_new[idx,:,:,:]
            batch_Y_selected = labels_new[idx]
      
            all_features = [(x, y) for x, y in zip(features_selected, batch_Y_selected)] #should shuffle
            print('random features:', features_selected.shape, batch_Y_selected.shape )   
            del features_new, labels_new
            
        for user in self.selected_users:  #change to selected users
            user.receive_features(all_features) ##class = id
            
        del all_features, features, labels, feature, label
        torch.cuda.empty_cache()

    def aggregate_features(self,glob_iter):  # all users or selected users
        assert (self.users is not None and len(self.users) > 0)
        history_round = 1   # include the current round
        remain_users = self.history_clients[-self.num_users*history_round:]
        current_users = [uid for uid in self.features.keys()]
        for uid in current_users:
            if uid not in remain_users:
                self.features.pop(uid)       

        for i, user in enumerate(self.selected_users):
            #if not user.shared:
            feature, label = user.send_features()   ##turned-off
            self.features[user.id] = [feature, label]
            del feature, label
        
        print('size of the global feature buffer:', len(self.features))
        torch.cuda.empty_cache()
            
    def sendto_features(self):
        for i, id in enumerate(self.features.keys()):
            if i == 0:
                features, labels = self.features[id]
            else:
                feature, label = self.features[id]
                features = torch.cat((features, feature), dim=0)
                labels = torch.cat((labels, label))
        all_features = [(x, y) for x, y in zip(features, labels)] 
        print('Current data/feature buff:', features.shape, labels.shape, len(all_features)) 
        labels = labels.to('cpu').tolist()
        print('Buffer classes:', Counter(labels))
        for user in self.selected_users:  #change to selected users
            user.receive_features(all_features) ##class = id
                   
    def send_aggragte_data(self,fre,fraction=1.0):  # all users or selected users
        assert (self.users is not None and len(self.users) > 0)
        for i, user in enumerate(self.selected_users):
            if i == 0:
                features, labels = user.send_data(fre,fraction)   
            else:
                feature, label = user.send_data(fre,fraction) 
                features = torch.cat((features, feature), dim=0)
                labels = torch.cat((labels, label))

        
        ##selected and mask y, for FedBR
        if self.algorithm == 'FedBR':
            labels = torch.ones(labels.shape)/10.0
            labels = labels.to(self.device)
            print(labels)
            np.random.seed(0)
            idx = np.random.choice(range(len(feature)), 32).tolist()  #only 32
            features = features[idx,:,:,:]
            labels = labels[idx,:]
        
        ## FedMix 
        all_features = [(x, y) for x, y in zip(features, labels)]  
        print('all aggragtion:', features.shape, labels.shape, len(all_features))   
            
        for user in self.selected_users:  #change to selected users
            if self.algorithm == 'Data_pre':
                user.receive_data(all_features) ## updata data loader
            else:
                user.receive_features(all_features)
            
        del all_features, features, labels, feature, label
        torch.cuda.empty_cache()        
             
    def aggregate_data(self,fre,fraction=1.0):  # selected users, fracation
        assert (self.users is not None and len(self.users) > 0)
        for i, user in enumerate(self.selected_users):
            if not user.shared:
                feature, label = user.send_data(fre,fraction)   ##turned-off
                self.features[user.id] = [feature, label]
           
    def sendto_data(self):        
        for i, id in enumerate(self.features.keys()):
            if i == 0:
                features, labels = self.features[id]
            else:
                feature, label = self.features[id]
                features = torch.cat((features, feature), dim=0)
                labels = torch.cat((labels, label))

        ##selected and mask y, for FedBR
        if self.algorithm == 'FedBR':
            labels = torch.ones(labels.shape)/10.0
            labels = labels.to(self.device)
            print(labels)
            np.random.seed(0)
            idx = np.random.choice(range(len(feature)), 32).tolist()  #only 32
            features = features[idx,:,:,:]
            labels = labels[idx,:]
        
        ## Data, FedMix, FedBR
        all_features = [(x, y) for x, y in zip(features, labels)] 
        print('Current data/feature buff:', features.shape, labels.shape, len(all_features))  
        ## Randomly sample
        idx = np.random.choice(range(len(all_features)), 500).tolist()
        # all_features = all_features[idx]
        # print('Shared data/feature buff:', features.shape, labels.shape, len(all_features))   
            
        for user in self.selected_users:  #change to selected users
            user.receive_features(all_features) ##class = id
            
        del all_features, features, labels, feature, label
        torch.cuda.empty_cache()

    def select_users(self, round, num_users, seed=0):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        np.random.seed(round + seed)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    def select_users_fea(self, num_users): #used for baseline
        return self.users[:num_users]
            