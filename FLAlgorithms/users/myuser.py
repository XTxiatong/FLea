import copy
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from FLAlgorithms.trainmodel.models import *
from FLAlgorithms.users.userbase import User
from torch import nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
import numpy as np


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

class Multilabel(nn.Module):
    def __init__(self):
        super(Multilabel, self).__init__()
    def forward(self, outputs, targets):
        logits = F.log_softmax(outputs)
        loss = - torch.sum(logits.mul(targets)) 
        return loss/outputs.shape[0]

class Corelation(nn.Module):
    def __init__(self):
        super(Corelation, self).__init__()
    def forward(self, data, feaure):
        n = data.shape[0]
        loss = dist_corr(data.reshape(n,-1),feaure.reshape(n,-1))
        return loss #0-1

class FedDecorrLoss(nn.Module):  #0.1

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            loss = torch.tensor([0.0]).to(x.device)
            return  loss

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss

class UserMain(User):   #model = model, model_name
    def __init__(self, numeric_id, train, test, test_ood, test_gb_ood, model, layer, percent, fea_dim, is_interpolated, 
                        batch_size, learning_rate, local_epochs, optimizer, 
                        personal_learning_rate, device, output_dim, loss='NLL', beta=1):
                                 
        super().__init__(numeric_id, train, test, test_ood, test_gb_ood, model, layer, percent, batch_size, learning_rate,
                         local_epochs, device, output_dim=output_dim,loss=loss,beta=beta)

        self.batch_size = batch_size
        self.N_Batch = len(train) // batch_size + 1
        self.lr = learning_rate
        self.plr = personal_learning_rate
        
        self.loss = loss
        self.device = device 
        self.fea_dim = fea_dim
        self.is_interpolated = is_interpolated
        
    def train(self, epochs, glob_iter):
        LOSS = 0
        N_Samples = 1
        Round = 1

        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        print('loss:', self.loss, 'learning rate:', self.plr*lr_decay)

        self.personal_model.train()
        self.global_model.eval()

        losses = torch.zeros(self.local_epochs) 
        losses2 = torch.zeros(self.local_epochs) 
       
        if self.loss == 'CE':  ##for baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
            print('training loss:', losses.mean()/self.N_Batch)

        if self.loss == 'CE_Prox': ##FedProx baseline
            '''
            from https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py
            '''
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    
                    mu = self.beta
                    proximal_term = 0.0
                    for w, w_t in zip(self.personal_model.parameters(), self.global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss += (mu / 2) * proximal_term
                    #print(loss.item(), (mu / 2) * proximal_term.item())
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    
            print('training loss:', losses.mean()/self.N_Batch)

        if self.loss == 'CE_Decorr': ##FEDDE baseline
            '''
            from https://github.com/bytedance/FedDecorr
            '''
            losses1 = torch.zeros(self.local_epochs)
            losses2 = torch.zeros(self.local_epochs)
        
            Loss = nn.CrossEntropyLoss(reduction='mean')
            Loss2 = FedDecorrLoss()
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    feature = self.personal_model.get_feature(batch_X, idx=17)
                    loss1 = Loss(logit,batch_Y)
                    loss2 = 0.1*Loss2(feature)
                    
                    losses1 [epoch] += loss1.item() 
                    losses2 [epoch] += loss2.item()
                    
                    loss = loss1 + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    
            print('training loss:', losses1.mean()/self.N_Batch, 'Cor:', losses2.mean()/self.N_Batch)
            
        if self.loss == 'CE_LC': ##FEDLC baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            def refine_loss(logits, targets):
                tau = self.beta
                num_classes = logits.shape[1]
                cla_fre = self.class_fre + 1e-6
                cal = cla_fre.repeat(logits.size(0), 1).to(logits.device)
                logits -= tau * cal**(-0.25)
                return logits

                # nt_positions = torch.arange(0, num_classes).to(logits.device)
                # nt_positions = nt_positions.repeat(logits.size(0), 1)
                # nt_positions = nt_positions[nt_positions[:, :] == targets.view(-1, 1)]
                # nt_positions = nt_positions.view(-1, 1)
                # t_logits = torch.gather(logits, 1, nt_positions)

                # t_logits = torch.exp(t_logits)
                # nt_logits = torch.exp(logits)
                # nt_logits = torch.sum(nt_logits, dim=1).view(-1,1) #- t_logits + 1e-6  #aviod inf
                
                # t_logits = torch.log(t_logits)
                # nt_logits = torch.log(nt_logits)
                
                # # print('t_logits', t_logits)
                # # print('nt_sum', nt_logits)
                # # print(t_logits - nt_logits)
                
                # #print(t_logits.shape, torch.sum(nt_logits, dim=1).shape)
                
                # loss = - t_logits + nt_logits
                # #print('loss:', loss)
                # return loss.mean()
 
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    logit = refine_loss(logit, batch_Y)
                    loss = Loss(logit,batch_Y)
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
            print('training loss:', losses.mean()/self.N_Batch)
            
        if self.loss == 'CE_RS': ##KDD FedRS baseline
            for epoch in range(self.local_epochs):
                batch_X,batch_Y = self.get_next_train_batch()
                output = self.personal_model(batch_X) #output logits
                rs_mask = torch.ones(output.shape).type(torch.float32).to(self.device)
                labels = torch.unique(batch_Y)
                for l in range(output.shape[1]):
                    if l not in labels:
                        rs_mask[:,l] = 0.5
                output = torch.mul(output,rs_mask)
                
                Loss = nn.CrossEntropyLoss()
                loss = Loss(output, batch_Y)
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.personal_model.parameters()),lr=self.plr*lr_decay) 
                self.optimizer.zero_grad()        
                loss.backward()
                nn.utils.clip_grad_norm_(self.personal_model.parameters(), 1) #gradient clip
                self.optimizer.step()
                losses[epoch] += loss.item()
            print('training loss:', losses.mean()/self.N_Batch)
            
        if self.loss == 'NT_CE': ## FedNTD baseline
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits
            Loss = nn.CrossEntropyLoss(reduction='mean')
            KLDiv = nn.KLDivLoss(reduction="batchmean")
            
            tau = 1
         
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    losses[epoch] += loss.item()
                    #print('CE loss:', loss.item())
                    
                    # Get smoothed local model prediction
                    logits = refine_as_not_true(logit, batch_Y)
                    pred_probs = F.log_softmax(logits / tau, dim=1)

                    # Get smoothed global model prediction
                    dg_logits = self.global_model(batch_X)  #remove no gred
                    dg_logits = refine_as_not_true(dg_logits, batch_Y)
                    dg_probs = F.softmax(dg_logits / tau, dim=1) ##note here torch.softmax is used

                    kl_loss = self.beta * (tau ** 2) * KLDiv(pred_probs, dg_probs)  ## CE loss with KL for not-true class
                    #print('KL loss:', kl_loss.item())
                    loss += kl_loss
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    
                    losses2[epoch] += kl_loss.item()
                    
            print('training loss:', losses.mean()/self.N_Batch, 'KL loss:', losses2.mean()/self.N_Batch)

            
        return LOSS

    def personal_train(self, epochs, glob_iter): 
        LOSS = 0
        N_Samples = 1
        Round = 1
        
        lr_decay = 1-glob_iter*0.005 if glob_iter < 100 else 0.5

        self.personal_model.train()
        self.global_model.eval()
        
        print('Personal model')
        test_acc = 0
        num = 0
        for data,label in self.testallloader:
            output = self.global_model(data) 
            test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
            num += len(data)
            del data, label, output
        print('Local Model Acc on testing data:', test_acc*1.0/num, 'length of data:', num)
        
        # ## save features
        # cnt = 0
        # for data,label in self.testallloader:
            # output = self.global_model.get_feature(data,idx=17)
            # prediction = self.global_model(data)
            # print(cnt)
            # features = torch.squeeze(output)
            # np.save(str(cnt)+'_test_feature.npy', features.cpu().detach())
            # np.save(str(cnt)+'_test_label.npy', label.cpu().detach())
            # np.save(str(cnt)+'_test_prediction.npy', prediction.cpu().detach())
            # cnt += 1
            
        # print('------------------------------------------------')

        losses = torch.zeros(self.local_epochs)#default 20
        
        if self.loss == 'CE':  ##for baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                print('------------------------------------------------')
                self.personal_model.train()
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    
                self.personal_model.eval()
                test_acc = 0
                num = 0
                for data,label in self.trainloader:
                    output = self.personal_model(data) 
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                    num += len(data)
                    del data, label, output
                print('Local Model Acc on training data:', test_acc*1.0/num, 'length of data:', num)
                
                test_acc = 0
                num = 0
                for data,label in self.testallloader:
                    output = self.personal_model(data) 
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                    num += len(data)
                    del data, label, output
                print('Local Model Acc on testing data:', test_acc*1.0/num, 'length of data:', num)
        
        print('training loss:', losses.mean()/self.N_Batch)
        
        
        self.personal_model.eval()
       
        cnt = 0
        for data,label in self.trainloader:
            output = self.personal_model.get_feature(data,idx=17)
            prediction = self.personal_model(data)
            print(cnt)
            features = torch.squeeze(output)
            np.save(str(cnt)+'_train_feature.npy', features.cpu().detach())
            np.save(str(cnt)+'_train_label.npy', label.cpu().detach())
            np.save(str(cnt)+'_train_prediction.npy', prediction.cpu().detach())
            cnt += 1
        
        ## save features
        cnt = 0
        for data,label in self.testallloader:
            output = self.personal_model.get_feature(data,idx=17)
            prediction = self.personal_model(data)
            print(cnt)
            features = torch.squeeze(output)
            np.save(str(cnt)+'_test_feature.npy', features.cpu().detach())
            np.save(str(cnt)+'_test_label.npy', label.cpu().detach())
            np.save(str(cnt)+'_test_prediction.npy', prediction.cpu().detach())
            cnt += 1
                 
    def post_train(self, epochs):   #used for baselines
  
        print('post-hoc retraining learning rate:', self.plr)
        print(self.modelname)
        
        self.personal_model.train()
        self.global_model.eval()
        
        # if self.modelname == 'MOBNET': ##Only update the final layer
            # for param in self.personal_model.conv1.parameters():
                # param.requires_grad = False
            # for param in self.personal_model.bn1.parameters():
                # param.requires_grad = False
            # for i in range(17):
                # for param in self.personal_model.layers[i].parameters():
                    # param.requires_grad = False    
            # for param in self.personal_model.conv2.parameters():
                # param.requires_grad = False
            # for param in self.personal_model.bn2.parameters():
                # param.requires_grad = False   


        if self.modelname == 'AudioNet':
            for param in self.personal_model.layers.parameters():
                param.requires_grad = False    
                
        losses = torch.zeros(self.local_epochs)
        if self.loss == 'CE':  ##for baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                for batch in range(self.feature_batch):
                    batch_X, batch_Y = self.get_next_feature_batch()
                    #print('batch feature:', batch_X.shape, batch_Y.shape)
                    
                    logit = self.personal_model(batch_X)
                    
                    loss = Loss(logit,batch_Y)
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr)
                    self.optimizer.zero_grad()        
                    loss.backward() #retain_graph=True
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    
                print('training loss:', losses[epoch].mean()/self.N_Batch)
                self.test('0')
        
    def train_Max_Min(self, epochs, glob_iter):  #baseline
        
        def get_unlabeled_by_self(all_x, all_y, all_unlabeled, q, K=1):
            one_hot_all_y = F.one_hot(all_y, q.shape[1]).to(all_x.device)
            new_q = torch.ones((len(all_y), q.shape[1])) / q.shape[1]
            new_q = new_q.to(all_x.device)
            for i in range(len(all_unlabeled)):
                for j in range(K):
                    index = torch.randint(0, len(all_x), (1,))
                    all_unlabeled[i] = all_unlabeled[i] + all_x[index]
                    new_q[i] = new_q[i] + one_hot_all_y[index]  #one hot
                all_unlabeled[i] = all_unlabeled[i] / (K + 1)
                new_q[i] = new_q[i] / (K + 1)

            return all_unlabeled, new_q  
        
        def sim(x1, x2):
            return torch.cosine_similarity(x1, x2, dim=1)
        
        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        print('Local adapation with distilling:', self.plr*lr_decay)
        
        losses =  torch.zeros(self.local_epochs)   
        losses1 = torch.zeros(self.local_epochs)
        losses2 = torch.zeros(self.local_epochs)
        losses3 = torch.zeros(self.local_epochs)

        self.personal_model.train()
        self.global_model.eval()
        self.discriminator.train()
        
        self.if_updated = True
        mu = 0.5
        lam = 0.8
        gamma = 1.0
        tau1 = 2.0
        tau2 = 2.0
        repeat = 1
        zeta = 1.5
        
        
        if self.loss == 'Max_Min':  ##CE and KL for global feature
            Loss = nn.CrossEntropyLoss(reduction='mean')

            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch() 
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_X = batch_X[:fea_num,]
                    batch_Y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    ## Train discriminator
                    all_unlabeled = batch_X_fea
                    q = torch.ones((len(batch_Y_fea), self.output_dim)) / self.output_dim
                    q = q.to(batch_X_fea.device)
                    all_unlabeled, q = get_unlabeled_by_self(batch_X, batch_Y, all_unlabeled, q)
        
                    if self.if_updated:
                        self.if_updated = False
                    
                    all_global_unlabeled_z = self.global_model.get_feature(all_unlabeled,17).clone().detach()
                    all_unlabeled_z = self.personal_model.get_feature(all_unlabeled,17)
                    all_self_z = self.personal_model.get_feature(batch_X,17)
                    #print('feature:', all_unlabeled_z.shape, all_self_z.shape)
                    
                    
                    
                    embedding1 = self.discriminator(all_unlabeled_z.clone().detach())  #
                    embedding2 = self.discriminator(all_global_unlabeled_z)
                    embedding3 = self.discriminator(all_self_z.clone().detach())

                    disc_loss = torch.log(torch.exp(sim(embedding1, embedding2) * tau1) / (torch.exp(sim(embedding1, embedding2) * tau1) + torch.exp(sim(embedding1, embedding3) * tau2)))
                    disc_loss = gamma * torch.sum(disc_loss) / len(embedding1)   #L_con

                    self.disc_opt= torch.optim.Adam(self.discriminator.parameters(), lr=self.plr*lr_decay)
                    self.disc_opt.zero_grad()
                    disc_loss.backward()
                    self.disc_opt.step()

                    
                    
                    ## train local model
                    
                    embedding1 = self.discriminator(all_unlabeled_z)
                    embedding2 = self.discriminator(all_global_unlabeled_z)
                    embedding3 = self.discriminator(all_self_z)

                    disc_loss = - torch.log(torch.exp(sim(embedding1, embedding2) * tau1) / (torch.exp(sim(embedding1, embedding2) * tau1) + torch.exp(sim(embedding1, embedding3) * tau2)))
                    disc_loss = torch.sum(disc_loss) / len(embedding1)

                    logit = self.personal_model(batch_X)
                    classifier_loss = Loss(logit,batch_Y) 
                    
                    logit_unlabel = self.personal_model(all_unlabeled)
                    aug_penalty = - torch.mean(torch.sum(torch.mul(F.log_softmax(logit_unlabel, 1), q), 1))
                    gen_loss =  classifier_loss + (mu * disc_loss) + lam * aug_penalty
       

                    self.gen_opt = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.gen_opt.zero_grad()  
                    self.disc_opt.zero_grad()
                    gen_loss.backward()
                    self.gen_opt.step()
                    
                    losses[epoch] += gen_loss.item()
                    losses1[epoch] += classifier_loss.item()
                    losses2[epoch] += aug_penalty.item()
                    losses3[epoch] += disc_loss.item()
                    
                    del gen_loss, aug_penalty, classifier_loss, disc_loss, logit, logit_unlabel, batch_X, batch_Y, batch_X_fea, batch_Y_fea
                    

             
        print(self.loss, 'Total:',  losses.mean()/self.N_Batch, 'local CE:', losses1.mean()/self.N_Batch, 'Feature CE:', losses2.mean()/self.N_Batch, 'disc_loss:', losses3.mean()/self.N_Batch) 
        
    def train_feature(self, epochs, glob_iter):  #used for baselines
  
        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        print('Local adapation with distilling:', self.plr*lr_decay)
         
        #self.model.train()
        self.personal_model.train()
        self.global_model.eval()
        
        losses1 = torch.zeros(self.local_epochs)
        losses2 = torch.zeros(self.local_epochs)
        
        if self.loss == 'CE_MCE':  ## FedMix 
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            Loss2 = Multilabel()
            print('data batch:', self.N_Batch, 'feature batch:', self.feature_batch)
            for epoch in range(self.local_epochs):
                #for batch in range(self.feature_batch): #feature data
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])

                    batch_x = batch_X[:fea_num,]
                    batch_y = batch_Y[:fea_num]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2)
                    #lam = 0.9  ## only local CE
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea
                    mix_y = lam*F.one_hot(batch_y,self.output_dim) + (1-lam)*batch_Y_fea
      
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global mixup CE
                    logit_lc = self.personal_model(mix_x)
                    loss2 = Loss2(logit_lc, mix_y)
                    
                    loss = loss1 + loss2
                    losses1[epoch] += loss1.item()
                    losses2[epoch] += loss2.item()
   
                  
                    loss_all = loss2 
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
        
        if self.loss == 'CE_CE':  ##for baseline 
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()

                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y) 

                    losses1[epoch] += loss.item()
                   
                    if self.layer >=0:
                        logit = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        loss2 = Loss(logit,batch_Y_fea)*self.beta
                    else:
                        logit = self.personal_model(batch_X_fea) #sharing raw data
                        loss2 = Loss(logit,batch_Y_fea)*self.beta
                        
 
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()

                    losses2[epoch] += loss2.item()
         
        if self.loss == 'CE_CE_KL':  ##CE and KL for global feature
            self.tau = 1
            print(self.tau)
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    #loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1 #+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
                    
        if self.loss == 'CE_CE_NT':   ##with FedNTD
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits
        
            Loss = nn.CrossEntropyLoss(reduction='mean')
            KLDiv = nn.KLDivLoss(reduction="batchmean")
            
            tau = 1
         
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y)
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_fe = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_fe_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_fe = self.personal_model(batch_X_fea) #sharing raw data
                        logit_fe_gb = self.global_model(batch_X_fea)
                         
                    loss2 = Loss(logit_fe,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
                    
                  
                    ## Global feature KL
                    logits_fe = refine_as_not_true(logit_fe, batch_Y_fea)
                    lc_probs_fe = F.log_softmax(logits_fe / tau, dim=1)
                    dg_logits_fe = refine_as_not_true(logit_fe_gb, batch_Y_fea)
                    dg_probs_fe = F.softmax(dg_logits_fe / tau, dim=1) ##note here torch.softmax is used
                    loss2 = self.beta * (tau ** 2) * KLDiv(lc_probs_fe, dg_probs_fe)
                    losses2[epoch] += loss2.item() 
                    
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                
                    del loss1, loss2, loss, loss_all, logit, logit_fe_gb, dg_logits_fe, dg_probs_fe, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_CE_localNT':   ##with FedNTD
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits
        
            Loss = nn.CrossEntropyLoss(reduction='mean')
            KLDiv = nn.KLDivLoss(reduction="batchmean")
            
            tau = 1
         
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y)
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_fe = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        #logit_fe_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_fe = self.personal_model(batch_X_fea) #sharing raw data
                        #logit_fe_gb = self.global_model(batch_X_fea)
                         
                    loss2 = Loss(logit_fe,batch_Y_fea)
                    loss = loss1 +loss2
                    losses1[epoch] += loss.item()
                    
                    ##  local data NT
                    logit_fe = logit
                    logit_fe_gb = self.global_model(batch_X)
                    logits_fe = refine_as_not_true(logit_fe, batch_Y)
                    lc_probs_fe = F.log_softmax(logits_fe / tau, dim=1)
                    dg_logits_fe = refine_as_not_true(logit_fe_gb, batch_Y)
                    dg_probs_fe = F.softmax(dg_logits_fe / tau, dim=1) ##note here torch.softmax is used
                    loss2 = self.beta * (tau ** 2) * KLDiv(lc_probs_fe, dg_probs_fe)
                    losses2[epoch] += loss2.item() 
                    
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                
                    del loss1, loss2, loss, loss_all, logit, logit_fe_gb, dg_logits_fe, dg_probs_fe, batch_X, batch_Y, batch_X_fea, batch_Y_fea
                                    
        if self.loss == 'CE_CE_KL_Prox':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## prox loss
                    mu = 0.01
                    proximal_term = 0.0
                    for w, w_t in zip(self.personal_model.parameters(), self.global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    proximal_term = (mu / 2) * proximal_term
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1 + loss2 + proximal_term
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2 
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea, proximal_term
        
        if self.loss == 'CE_CE_KL_LC':  ##CE and KL for global feature, plus LC for local data
            def refine_loss(logits, targets):
                #tau = self.beta
                num_classes = logits.shape[1]
                cla_fre = self.class_fre + 1e-6
                cal = cla_fre.repeat(logits.size(0), 1).to(logits.device)
                logits -= 1.0 * cal**(-0.25)
                return logits
                
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    logit = refine_loss(logit, batch_Y)
                    loss1 = Loss(logit,batch_Y) 

                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2 
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea
 
        if self.loss == 'CE_Mix_KL':  ##CE and KL for global feature, mixup local data and global faeture
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')

            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): ##feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_x = self.extract_features(batch_X) ##local features
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  ##在|feature|<|X|时，可通过映入随机性，提高性能
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_x[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2)
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea

                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global feature CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(mix_x, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(mix_x, idx=self.layer)
                    else:
                        logit_gb = self.global_model(mix_x)
                        logit_lc = self.personal_model(mix_x)

                    # print('----------')
                    loss2 = lam*Loss(logit_lc,batch_y) + (1-lam)*Loss(logit_lc,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_Mix_NT':  ##CE and KL for global feature, mixup local data and global faeture, distilling from global feature
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits

            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')

            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): ##feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_x = self.extract_features(batch_X) ##local features
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  ##在|feature|<|X|时，可通过映入随机性，提高性能
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_x[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2)
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea

                    ## local data CE
                    logit = self.personal_model(batch_X)
                    logit_gb = self.global_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(mix_x, idx=self.layer)
                        logit_lc2 = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb2 = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        
                        logit_lc = self.personal_model(mix_x)
                        logit_lc2 = self.personal_model(batch_X_fea)
                        logit_gb2 = self.global_model(batch_X_fea)

                    ## global feature CE
                    loss2 = lam*Loss(logit_lc,batch_y) + (1-lam)*Loss(logit_lc,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## all data distillation, not true class distilling
                    batch_Y = torch.cat((batch_Y,batch_Y_fea),dim=0)
                    logit_gb = torch.cat((logit_gb,logit_gb2),dim=0)
                    logit_lc = torch.cat((logit,logit_lc2),dim=0)
                    logit_gb = refine_as_not_true(logit_gb, batch_Y)
                    logit_lc = refine_as_not_true(logit_lc, batch_Y)
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_MCE_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            Loss2 = Multilabel()
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): ##feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_x = self.extract_features(batch_X) ##local features
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  ##在|feature|<|X|时，可通过映入随机性，提高性能
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_x[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2)
                    #lam = np.random.beta(2, 2, fea_num)
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea
                    print(mix_x.shape, batch_x.shape)
                    mix_y = lam*F.one_hot(batch_y,self.output_dim) + (1-lam)*F.one_hot(batch_Y_fea,self.output_dim)

                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global feature CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(mix_x, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(mix_x, idx=self.layer)
                    else:
                        logit_gb = self.global_model(mix_x)
                        logit_lc = self.personal_model(mix_x)

                    #print('----------')
                    #print(lam*Loss(logit_lc,batch_y) + (1-lam)*Loss(logit_lc,batch_Y_fea))
                    loss2 = Loss2(logit_lc, mix_y)
                    #print(loss2)
                    
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_MCE2_KL':  ##CE and KL for global feature
            self.tau = 1
            self.beta = 0.01
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            Loss2 = Multilabel()
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): ##feature data
                    batch_X, batch_Y = self.get_next_train_batch() 
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  ##data 

                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global feature CE
                    logit_gb = self.global_model(batch_X_fea)
                    logit_lc = self.personal_model(batch_X_fea)
                    loss2 = Loss2(logit_lc, batch_Y_fea) 
                    #print(loss1.item(), loss2.item())
                    loss = loss1 +loss2*0.01
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                    #print(loss1.item(), loss2.item())
                   
                    loss_all = loss #+ loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_Cor_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            correlation = []
            
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer)
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ##local feature decorrelation
                    loss2 = CorLoss(batch_X, batch_F)
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss3 = Loss(logit_lc,batch_Y_fea)

                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss4 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    
                    loss_all = loss1*0.8 + loss2 + loss3 + loss4
                    
                    losses1[epoch] += loss_all.item()
                    losses2[epoch] += loss2.item()
                    correlation.append(loss2.item())
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss3, loss4, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
                    
            print('decorrelation:', losses2/self.N_Batch) 
           
        print(self.loss, 'training loss:', losses1.mean()/self.N_Batch, 'feature loss:', losses2.mean()/self.N_Batch)      
        
    def train_decor(self, epochs, glob_iter, lossname=None):  #used for FLea warmup
  
        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        print('Local adapation with distilling:', self.plr*lr_decay)
         
        self.personal_model.train()
        self.global_model.eval()
        
        self.tau = 1
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        CELoss = nn.CrossEntropyLoss(reduction='mean')
        CorLoss = Corelation()
        
        losses1 = torch.zeros(epochs)
        losses2 = torch.zeros(epochs)
        losses3 = torch.zeros(epochs)
        
        alpha3 = 3
        print('alpha3:', alpha3)
        for epoch in range(epochs):
            for batch in range(self.N_Batch): #feature data
                batch_X, batch_Y = self.get_next_train_batch()
                batch_F = self.personal_model.get_feature(batch_X, idx=self.layer) #with gradience
              
                ## local data CE
                logit = self.personal_model(batch_X)
                loss1 = CELoss(logit,batch_Y) 
                

                ## local data distilling
                logit_gb = self.global_model(batch_X)
                pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                pro_lc = F.log_softmax(logit / self.tau, dim=1) ## x 
                loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                losses2[epoch] += loss2.item()
                
                ##local feature decorrelation
                
                loss3 = CorLoss(batch_X, batch_F) 
                losses3[epoch] += loss3.item()

                if lossname == 'CE_KL':
                    loss_all = loss1 + loss2
                    losses1[epoch] += loss1.item()
                
                elif lossname == 'COR_KL':
                    loss_all = loss2 +loss3
                    losses1[epoch] += loss1.item()
                    
                elif lossname == 'CE_COR_KL':
                    #loss_all = loss1 + loss2*0.5 + loss3*5
                    loss_all = loss1 + loss2*0.5 + loss3*alpha3 #self.alpha_3=3
                    losses1[epoch] += loss1.item()


                self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                self.optimizer.zero_grad()        
                loss_all.backward()  #retain_graph=True
                self.optimizer.step()
                
        print('training:',lossname, 'CEloss:', losses1.mean()/self.N_Batch, 'KLloss:', losses2.mean()/self.N_Batch, 'Decorrelation loss:', losses3.mean()*alpha3/self.N_Batch)
        print('decorrelation:', losses3/self.N_Batch)
        print('The last decorrelation:', losses3[epochs-1]/self.N_Batch)

    def train_decor_feature(self, epochs, glob_iter):  #used for FLea
  
        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        #self.alpha_3 =  0.5 - glob_iter*0.1 if glob_iter < 5 else 0
        #self.alpha_3 =  1 - glob_iter*0.05 if glob_iter < 18 else 0.05
        self.alpha_3 =  3 - glob_iter*0.1 if glob_iter < 20 else 1 
        #self.alpha_3 =  8 - glob_iter*0.1 if glob_iter < 70 else 1
        #self.alpha_3 =  10 - glob_iter*0.1 if glob_iter < 90 else 1 
        #self.alpha_4 = 6
        print('Local adapation with distilling:', self.plr*lr_decay, 'alphas:', self.alpha_1,self.alpha_2,self.alpha_3,self.alpha_4)
         
        #self.model.train()
        self.personal_model.train()
        self.global_model.eval()
        
        losses =  torch.zeros(self.local_epochs)   
        losses1 = torch.zeros(self.local_epochs)
        losses2 = torch.zeros(self.local_epochs)
        losses3 = torch.zeros(self.local_epochs)
        losses4 = torch.zeros(self.local_epochs)

        if self.loss == 'CE_CE_DeC_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer)
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y)

                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_lc = self.personal_model(batch_X_fea)
                        logit_gb = self.global_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)

                    ##local feature decorrelation
                    loss3 = CorLoss(batch_X, batch_F)
                    
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss4 = KLLoss(pro_lc,pro_gb)*(self.tau ** 2)
                    
                    loss_all = loss1*self.alpha_1 + loss2*self.alpha_2 + loss3*self.alpha_3 + loss4*self.alpha_4
                    
                    losses1[epoch] += loss1.item()
                    losses2[epoch] += loss2.item()
                    losses3[epoch] += loss3.item()
                    losses4[epoch] += loss4.item()
                    losses[epoch] += loss_all.item() 
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss3, loss4, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
            print(self.loss, 'Total:',  losses.mean()/self.N_Batch, 'local CE:', losses1.mean()/self.N_Batch, 'Feature CE:', losses2.mean()/self.N_Batch, 'KL:', losses4.mean()/self.N_Batch) 
            print('The last decorrelation:', losses3[epochs-1]/self.N_Batch)
        
        if self.loss == 'CE_CE_DeC_NT':  ##CE and KL for global feature
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits
            tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            
            
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer)
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)

                    ##local feature decorrelation
                    loss3 = CorLoss(batch_X, batch_F)
                    
                    ## global data distilling
                    logits_fe = refine_as_not_true(logit_lc, batch_Y_fea)
                    lc_probs_fe = F.log_softmax(logits_fe / tau, dim=1)
                    dg_logits_fe = refine_as_not_true(logit_gb, batch_Y_fea)
                    dg_probs_fe = F.softmax(dg_logits_fe / tau, dim=1) ##note here torch.softmax is used
                    loss4 = (tau ** 2) * KLLoss(lc_probs_fe, dg_probs_fe)
                    
                    loss_all = loss1*self.alpha_1 + loss2*self.alpha_2 + loss3*self.alpha_3 + loss4*self.alpha_4
                    
                    losses1[epoch] += loss1.item()*self.alpha_1
                    losses2[epoch] += loss2.item()*self.alpha_2
                    losses3[epoch] += loss3.item()*self.alpha_3
                    losses4[epoch] += loss4.item()*self.alpha_4
                    losses[epoch] += loss_all.item() 
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss3, loss4, loss_all, logit, logit_gb, logit_lc, batch_X, batch                    

            print('The last decorrelation:', losses4[self.local_epochs-1]/self.N_Batch)

        if self.loss == 'CE_MCE_DeC_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            MCELoss = Multilabel() 
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_x = self.extract_features(batch_X) ##local features
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer)
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_x[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2)
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea
                    mix_y = lam*F.one_hot(batch_y,self.output_dim) + (1-lam)*F.one_hot(batch_Y_fea,self.output_dim)
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    mix_logit = self.personal_model.forward_feature(mix_x, idx=self.layer)
                    loss2 = MCELoss(mix_logit,mix_y)

                    ##local feature decorrelation
                    loss3 = CorLoss(batch_X, batch_F)
                    
                    ## global data distilling
                    logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                    logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer) 
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss4 = KLLoss(pro_lc,pro_gb)*(self.tau ** 2)
                    
                    loss_all = loss1*self.alpha_1 + loss2*self.alpha_2 + loss3*self.alpha_3 + loss4*self.alpha_4
                    
                    losses1[epoch] += loss1.item() 
                    losses2[epoch] += loss2.item() 
                    losses3[epoch] += loss3.item() 
                    losses4[epoch] += loss4.item() 
                    losses[epoch] += loss_all.item() 
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss3, loss4, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
             
            print(self.loss, 'Total:',  losses.mean()/self.N_Batch, 'local CE:', losses1.mean()/self.N_Batch, 'Feature CE:', losses2.mean()/self.N_Batch, 'KL:', losses4.mean()/self.N_Batch) 
            print('The last decorrelation:', losses3[epochs-1]/self.N_Batch)

        if self.loss == 'MCE_DeC_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            MCELoss = Multilabel() 
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    #batch_x = self.extract_features(batch_X) ##local features
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer)
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_F[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2,fea_num)
                    mix_x = torch.zeros(batch_x.shape).to(self.device)
                    mix_y = torch.zeros(F.one_hot(batch_y,self.output_dim).shape).to(self.device)
                    for i in range(fea_num):
                        mix_x[i,:,:,:] = lam[i]*batch_x[i,:,:,:] + (1-lam[i])*batch_X_fea[i,:,:,:]
                        mix_y[i,:] = lam[i]*F.one_hot(batch_y,self.output_dim)[i,:] + (1-lam[i])*F.one_hot(batch_Y_fea,self.output_dim)[i,:]
                    #mix_x = lam*batch_x + (1-lam)*batch_X_fea
                    #mix_y = lam*F.one_hot(batch_y,self.output_dim) + (1-lam)*F.one_hot(batch_Y_fea,self.output_dim)
                    # print('lam:', batch, lam)
                    # print('mix_y:', batch, mix_y)
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    mix_logit = self.personal_model.forward_feature(mix_x, idx=self.layer)
                    loss2 = MCELoss(mix_logit,mix_y)

                    ##local feature decorrelation
                    loss3 = CorLoss(batch_X, batch_F)
                    
                    ## global data distilling
                    logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                    logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer) 
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss4 = KLLoss(pro_lc,pro_gb)*(self.tau ** 2)
                    
                    loss_all = loss1*self.alpha_1 + loss2*self.alpha_2 + loss3*self.alpha_3 + loss4*self.alpha_4
                    
                    losses1[epoch] += loss1.item() 
                    losses2[epoch] += loss2.item() 
                    losses3[epoch] += loss3.item() 
                    losses4[epoch] += loss4.item() 
                    losses[epoch] += loss_all.item() 
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss3, loss4, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
             
            print(self.loss, 'Total:',  losses.mean()/self.N_Batch, 'local CE:', losses1.mean()/self.N_Batch, 'Feature CE:', losses2.mean()/self.N_Batch, 'KL:', losses4.mean()/self.N_Batch) 
            print('The last decorrelation:', losses3[epochs-1]/self.N_Batch)
            
        if self.loss == 'CE_MCE_DeC':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            MCELoss = Multilabel() 
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_x = self.extract_features(batch_X) ##local features
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer)
                    
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_x[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                    
                    lam = np.random.beta(2,2)
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea
                    mix_y = lam*F.one_hot(batch_y,self.output_dim) + (1-lam)*F.one_hot(batch_Y_fea,self.output_dim)
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    mix_logit = self.personal_model.forward_feature(mix_x, idx=self.layer)
                    loss2 = MCELoss(mix_logit,mix_y)

                    ##local feature decorrelation
                    loss3 = CorLoss(batch_X, batch_F)
                    
                    # ## global data distilling
                    # logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                    # logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer) 
                    # pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    # pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # # The smallest KL is 0, always positive 
                    # loss4 = KLLoss(pro_lc,pro_gb)*(self.tau ** 2)
                    
                    loss_all = loss1*self.alpha_1 + loss2*self.alpha_2 + loss3*self.alpha_3 #+ loss4*self.alpha_4
                    
                    losses1[epoch] += loss1.item() 
                    losses2[epoch] += loss2.item() 
                    losses3[epoch] += loss3.item() 
                    #losses4[epoch] += loss4.item() 
                    losses[epoch] += loss_all.item() 
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss3, loss_all, logit, batch_X, batch_Y, batch_X_fea, batch_Y_fea,mix_x,mix_y
             
            print(self.loss, 'Total:',  losses.mean()/self.N_Batch, 'local CE:', losses1.mean()/self.N_Batch, 'Feature CE:', losses2.mean()/self.N_Batch) 
            print('The last decorrelation:', losses3[epochs-1]/self.N_Batch)

        if self.loss == 'CE_DeC_KL_MCE':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            MCELoss = Multilabel() 
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()  
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ##local feature decorrelation
                    batch_F = self.personal_model.get_feature(batch_X, idx=self.layer) #with gradience
                    loss3 = CorLoss(batch_X, batch_F)
                    
                    ## global data distilling
                    logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                    logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer) 
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss4 = KLLoss(pro_lc,pro_gb)*(self.tau ** 2)
                    
                    loss_all = loss1*self.alpha_1 + loss3*self.alpha_3 + loss4*self.alpha_4

                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    ## global data CE
                    batch_x = self.extract_features(batch_X) #no gradience
                    fea_num = min(batch_X.shape[0], batch_X_fea.shape[0])
                    batch_x = batch_x[:fea_num,]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:fea_num,]
                    batch_Y_fea = batch_Y_fea[:fea_num,]
                   
                    #lam = np.random.beta(2,2)
                    lam = 0.5
                    mix_x = lam*batch_x + (1-lam)*batch_X_fea
                    mix_y = lam*F.one_hot(batch_y,self.output_dim) + (1-lam)*F.one_hot(batch_Y_fea,self.output_dim)
                    
                    mix_logit = self.personal_model.forward_feature(mix_x, idx=self.layer)
                    loss2 = MCELoss(mix_logit,mix_y)
                    loss_all = loss2*self.alpha_2

                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #The whole model
                    self.optimizer.step()

                    losses1[epoch] += loss1.item() 
                    losses2[epoch] += loss2.item() 
                    losses3[epoch] += loss3.item() 
                    losses4[epoch] += loss4.item() 
                    losses[epoch] += loss_all.item() 

                    del loss1, loss2, loss3, loss4, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
             
            print(self.loss, 'Total:',  losses.mean()/self.N_Batch, 'local CE:', losses1.mean()/self.N_Batch, 'Feature CE:', losses2.mean()/self.N_Batch, 'KL:', losses4.mean()/self.N_Batch) 
            print('The last decorrelation:', losses3[epochs-1]/self.N_Batch)

