import json
import os
import torch



def load_data(train_data, test_data):
    
    train_client_data = train_data['user_data']
    test_client_data = test_data['user_data']
    test_global = test_data['global_test'] 
    
    clients = list(sorted(train_client_data.keys())) ##aviod shuffle 
    return clients, train_client_data, test_client_data, test_global
    

def read_user_data(index, data, dataset, device=None):
    
    id = data[0][index]
    train_data = data[1][id]
    test_data = data[2][id]
    global_test = data[3]
    
    device = torch.device('cpu') if device is None else device
    
    if dataset == "Mnist":
        IMAGE_SIZE = 28
        IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
        NUM_CHANNELS = 1
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        #X_ood, y_ood = ood_data['x'], ood_data['y'],
        X_global, y_global =  global_test['x'], global_test['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS*IMAGE_SIZE*IMAGE_SIZE).type(torch.float32).to(device) 
        y_train = torch.Tensor(y_train).type(torch.int64).to(device)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS*IMAGE_SIZE*IMAGE_SIZE).type(torch.float32).to(device)
        y_test = torch.Tensor(y_test).type(torch.int64).to(device)
        #X_ood = torch.Tensor(X_ood).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32).to(device)
        #y_ood = torch.Tensor(y_ood).type(torch.int64).to(device)
        X_global = torch.Tensor(X_global).view(-1, NUM_CHANNELS*IMAGE_SIZE*IMAGE_SIZE).type(torch.float32).to(device)
        y_global = torch.Tensor(y_global).type(torch.int64).to(device)
    elif dataset =='Cifar10':
        IMAGE_SIZE = 32
        NUM_CHANNELS = 3
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_global, y_global =  global_test['x'], global_test['y']
        
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32).to(device)   
        y_train = torch.Tensor(y_train).type(torch.int64).to(device)
        
        
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32).to(device)  
        y_test = torch.Tensor(y_test).type(torch.int64).to(device)
        
        X_global = torch.Tensor(X_global).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32).to(device)
        y_global = torch.Tensor(y_global).type(torch.int64).to(device)

    elif dataset == 'UrbanSound':
    
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_global, y_global =  global_test['x'], global_test['y']

        X_train = torch.stack(X_train).type(torch.float32).to(device)
        y_train = torch.Tensor(y_train).type(torch.int64).to(device)
        X_test = torch.stack(X_test).type(torch.float32).to(device)
        y_test = torch.Tensor(y_test).type(torch.int64).to(device)
        X_global = torch.stack(X_global).type(torch.float32).to(device)
        y_global = torch.Tensor(y_global).type(torch.int64).to(device)
        
        print(X_train.shape, y_train.shape, X_global.shape)

    else:
        
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_global, y_global =  global_test['x'], global_test['y']
    
        X_train = torch.Tensor(X_train).view(-1,1,128,3).type(torch.float32).to(device)
        y_train = torch.Tensor(y_train).type(torch.int64).to(device)
        X_test = torch.Tensor(X_test).view(-1,1,128,3).type(torch.float32).to(device)
        y_test = torch.Tensor(y_test).type(torch.int64).to(device)
        X_global = torch.Tensor(X_global).view(-1,1,128,3).type(torch.float32).to(device)
        y_global = torch.Tensor(y_global).type(torch.int64).to(device)

        print(X_train.shape, y_train.shape, X_global.shape)
        
        
        

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    #test_ood = [(x, y) for x, y in zip(X_ood, y_ood)]
    test_global = [(x, y) for x, y in zip(X_global, y_global)]
    test_ood = []
    return id, train_data, test_data, test_ood, test_global


def read_data(dataset, subset='data'):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    print(dataset)
    train_data_dir = os.path.join('data', dataset, subset, 'train')
    test_data_dir = os.path.join('data', dataset, subset, 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    test_ood = {}
    test_global = {} 

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files: #only one json
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data']) #local testing
        test_ood.update(cdata['ood_data'])  #local ood
        test_global.update(cdata['global_set']) ##global set
        
        
    clients = list(sorted(train_data.keys()))
    print('train data num:\n', [len(x['y']) for x in train_data.values()])
    return clients, groups, train_data, test_data, test_ood, test_global