#!/usr/bin/env python
from FLAlgorithms.servers.myserver import ServerMain
from FLAlgorithms.trainmodel.models import *
import argparse
#from utils.data_loader import *
from utils.model_utils import read_data, load_data
from utils.data_utils import split_data
import torch
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.stdout.flush()

torch.manual_seed(1)
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True



def main(dataset, class_num, split_method, split_para, split_num, algorithm, modelname, loss, 
         batch_size, learning_rate, num_glob_iters, layer, fea_percent, fea_dim, is_interpolated, 
         local_epochs, optimizer, client_num, personal_learning_rate, device, beta,mode, seed):
    post_fix_str = '{}_{}_{}_loss_{}_epoch_{}_{}_client_{}_split_{}_{}'.format(algorithm, modelname, dataset,loss, local_epochs,num_glob_iters, split_num, split_method, split_para)
    model_path = []
    
    train_data, test_data = split_data(dataset, class_num, split_num, split_method, split_para)
    data = load_data(train_data, test_data) #for testing
    
    
    if dataset == 'Cifar10':
        inputdim = 32*32
        if modelname == 'VGG':
            model = VGG16(class_num).to(device), modelname
        elif modelname == 'RESNET':
            model = ResNet18(class_num).to(device), modelname
        elif modelname == 'VIT':
            model = VisionTransformer(class_num).to(device), modelname
        elif modelname == 'MOBNET':
            model = MobileNetV2(class_num).to(device), modelname
    elif dataset == 'UrbanSound': 
        model = AudioClassifier(class_num).to(device), modelname      
    elif dataset == 'UCI_HAR': 
        model = HARNet(class_num).to(device), modelname
        
    print(model[0])
    print('#######', device, '#######' )
    pytorch_total_params = sum(p.numel() for p in model[0].parameters())
    print('number of parameters:', pytorch_total_params)

    for i in range(1): #repeat times
        if mode == 'personalisation': # for personlisaton test
            model_path = os.path.join("models", 'test')  
            #checkpoint_path = os.path.join(model_path, "server_FedAvg_MOBNET_Cifar10_loss_CE_epoch_10_100_client_1000_split_quantity_10.0.pt")
            #checkpoint_path = os.path.join(model_path, "server_FedAvg_MOBNET_Cifar10_loss_CE_epoch_10_100_client_10_split_quantity_10.0.pt")
            checkpoint_path = os.path.join(model_path, "server_FLea_MOBNET_Cifar10_loss_CE_CE_DeC_KL_epoch_10_100_client_100_split_quantity_3.0.pt")
            checkpoint= torch.load(checkpoint_path)
            print('Load model checkpoint from name succuessfully!') 
            
            model = checkpoint, modelname
            server = ServerMain(dataset, data, algorithm, model, batch_size, learning_rate, num_glob_iters, layer, fea_percent, fea_dim, is_interpolated, 
                             local_epochs, optimizer, client_num, i, device, personal_learning_rate,
                             output_dim=class_num, post_fix_str=post_fix_str, loss=loss, beta=beta, seed=seed)
            del data
            del train_data, test_data  
            server.personal_train()
           
        else: #Training and test
            server = ServerMain(dataset, data, algorithm, model, batch_size, learning_rate, num_glob_iters, layer, fea_percent, fea_dim, is_interpolated, 
                             local_epochs, optimizer, client_num, i, device, personal_learning_rate,
                             output_dim=class_num, post_fix_str=post_fix_str, loss=loss, beta=beta, seed=seed)

            del data
            del train_data, test_data   
            server.train()



def run():

    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--class_num", type=int, default=10)
    
    parser.add_argument("--split_method", type=str, default='quantity') #distribution
    parser.add_argument("--split_para", type=float, default=5.0) #for split it is #preesnt class, for lda, it is \alpha
    parser.add_argument("--split_num", type=int, default=10)
    
    parser.add_argument("--client_num", type=int, default=10) # total clients, should be <= split_num
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--fea_percent", type=float, default=0.1)
    parser.add_argument("--fea_dim", type=int, default=64)
    parser.add_argument("--is_interpolated", type=bool, default=False)
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001) #not used
    parser.add_argument("--personal_learning_rate", type=float, default=0.001)                    
    parser.add_argument("--num_global_iters", type=int, default=500)
    parser.add_argument("--local_epochs", type=int, default=10)    
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--beta", type=float, default=1.0)
 
    parser.add_argument("--seed", type=int, default=0, help="seed for client selection")
    parser.add_argument("--mode", type=str, default="training")
    parser.add_argument("--modelname", type=str, default="MOBNET")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("Summary of training process:") 
    print("Dataset                : {}".format(args.dataset))
    print("Batch size             : {}".format(args.batch_size))
    print("Learing rate           : {}".format(args.personal_learning_rate))
    print("Number of total clients: {}".format(args.split_num))
    print("Split method           : {}".format(args.split_method))
    print("Split parameter        : {}".format(args.split_para))
    print("Clients per round      : {}".format(args.client_num))
    print("Number of global rounds: {}".format(args.num_global_iters))
    print("Number of local rounds : {}".format(args.local_epochs))
    print("Feature from layer     : {}".format(args.layer))
    print("Feature percentage     : {}".format(args.fea_percent))
    print("Is interplolated       : {}".format(args.is_interpolated))
    print("Local training loss    : {}".format(args.loss))
    print("Loss of beta           : {}".format(args.beta))
    print("Algorithm              : {}".format(args.algorithm))
    print("Modelname              : {}".format(args.modelname))
    print("Mode                   : {}".format(args.mode))
    print("Seed                   : {}".format(args.seed))
    print("=" * 80)

    return main(
        dataset=args.dataset,
        class_num=args.class_num,
        split_method=args.split_method,
        split_para=args.split_para,
        split_num=args.split_num,
        algorithm=args.algorithm,
        modelname=args.modelname,
        loss=args.loss,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_glob_iters=args.num_global_iters,
        layer=args.layer,
        fea_percent=args.fea_percent,
        fea_dim=args.fea_dim, 
        is_interpolated=args.is_interpolated,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        client_num=args.client_num,
        personal_learning_rate=args.personal_learning_rate,
        device=device,
        beta=args.beta,
        mode=args.mode,
        seed=args.seed
        
    )

     
 

if __name__ == "__main__":
    run()
 
