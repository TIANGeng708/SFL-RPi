#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
import torch
import requests
import time
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame

import random
import numpy as np
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "SFLV1 ResNet18 on HAM10000"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

#===================================================================
# No. of users
num_users = 1
epochs = 200
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.01


#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 32 * 7 * 7)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x
net_glob_client = CNN()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(
        net_glob_client)  # to use the multiple GPUs; later we can change this to CPUs only

net_glob_client.to(device)
print(net_glob_client)


# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side




# net_glob_server = ResNet18_server_side(ResidualBlock)  # 7 is my numbr of classes
# if torch.cuda.device_count() > 1:
#     print("We use", torch.cuda.device_count(), "GPUs")
#     net_glob_server = nn.DataParallel(net_glob_server)  # to use the multiple GPUs
#
# net_glob_server.to(device)
# print(net_glob_server)

# ===================================================================================
# For Server Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
run_time=[]

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0
train_count=0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# w_glob_server = net_glob_server.state_dict()
# w_locals_server = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
# net_model_server = [net_glob_server for i in range(num_users)]
# net_server = copy.deepcopy(net_model_server[0]).to(device)
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)


#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 5
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 100, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 100, shuffle = True)
        

    def train(self, net):
        global train_count
        net.train()
        optimizer_client = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                torch.save(client_fx, 't3_send1.pt')
                torch.save(labels, 't3_send2.pt')
                with open('t3_client.txt', 'wb') as file:
                    file.write(str(train_count).encode())

                print(train_count)
                # dfx = split_model_server.train_server(client_fx, labels, iter, self.local_ep, self.idx, 1)
                while True:
                    print('a')
                    try:
                        print('b')
                        r2 = requests.get('http://192.168.1.107:8003/s3/t.txt', timeout=5)
                    except:
                        print('c')
                        continue
                    if r2.content.decode() == str(train_count + 1):
                        break
                    time.sleep(10)
                print('d')
                r1 = 0
                while True:
                    print('e')
                    try:
                        print('f')
                        r1 = requests.get('http://192.168.1.107:8003/s3/server_send.pt', timeout=5)
                    except:
                        print('g')
                        continue
                    break
                print('h')
                with open('dfx3.pt', 'wb') as file:
                    file.write(r1.content)
                dfx = torch.load('dfx3.pt')
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                train_count += 1
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    #
    # def evaluate(self, net, ell):
    #     net.eval()
    #
    #     with torch.no_grad():
    #         len_batch = len(self.ldr_test)
    #         for batch_idx, (images, labels) in enumerate(self.ldr_test):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             #---------forward prop-------------
    #             fx = net(images)
    #
    #             # Sending activations to server
    #             evaluate_server(fx, labels, self.idx, len_batch, ell)
    #
    #         #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
    #
    #     return
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this

def cifar_user_dataset(dataset, num_users, noniid_fraction):
    """
    Sample a 'fraction' of non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param fraction:
    :return:
    """

    # initialization
    total_items = len(dataset)
    num_noniid_items = len(dataset) * noniid_fraction
    num_iid_items = total_items - num_noniid_items
    dict_users = list()
    for ii in range(num_users):
        dict_users.append(list())
    idxs = [i for i in range(len(dataset))]

    # IID
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users)
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += tmp_set
            idxs = list(set(idxs) - tmp_set)

    # NON-IID
    if num_noniid_items != 0:

        num_shards = num_users  # each user has one shard
        per_shards_num_imgs = int(num_noniid_items / num_shards)
        idx_shard = [i for i in range(num_shards)]
        labels = list()
        for ii in range(len(idxs)):
            labels.append(dataset[idxs[ii]][1])
        print(labels)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # for i in range(len(idxs_labels)):
        #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #     print(idxs_labels[i])
        idxs = idxs_labels[0, :]

        # divide and assign
        i = 0
        while idx_shard:
            print(idx_shard)
            rand_idx = np.random.choice(idx_shard, 1, replace=False)
            rand_idx[0] = idx_shard[0]
            # rand_idx.append(idx_shard[0])
            print(rand_idx)
            idx_shard = list(set(idx_shard) - set(rand_idx))
            dict_users[i].extend(idxs[int(rand_idx) * per_shards_num_imgs: (int(rand_idx) + 1) * per_shards_num_imgs])
            i = divmod(i + 1, num_users)[1]

    '''
    for ii in range(num_users):
        tmp = list()
        for jj in range(len(dict_users[ii])):
            tmp.append(dataset[dict_users[ii][jj]][1])
        tmp.sort()
        print(tmp)
    '''
    return dict_users

def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          
#=============================================================================
#                         Data loading 
#============================================================================= 

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)




# ans = []
# for testImgs, labels in dataset_train:
#     testImgs = testImgs
#     labels = labels
#     outputs = model(testImgs)
#     for element in outputs:
#         tmp = element.detach().numpy()
#         ans.append(np.argmax(tmp))
#
#         target = torch.tensor(ans)
dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
# dataset_train = SkinData(train, transform = train_transforms)
# dataset_test = SkinData(test, transform = test_transforms)

# ----------------------------------------------------------------
# with open('beta=0.1.pkl', 'rb') as file:
#     dict_users=pickle.load(file)
# dict_users=cifar_user_dataset(dataset_train,num_users,0)
dict_users=[[]]
for i in range(18000,24000):
    dict_users[0].append(i)
#dict_users_test=cifar_user_dataset(dataset_test,num_users,0.9)
# dict_users_test = dataset_iid(dataset_test, num_users)
dict_users_test=[[]]
for i in range(5000):
    dict_users_test[0].append(i)

#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []
      
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        
        # Testing -------------------
        # local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        
            
    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    torch.save(net_glob_client.state_dict(), 'avg.pt')
    with open('avg.txt', 'wb') as file:
        file.write('+'.encode())
    while True:
        try:
            avg = requests.get('http://192.168.1.107:8001/s3/avg_c_read.txt', timeout=5)
        except:
            continue
        if avg.content.decode() == '+':
            try:
                avg_new = requests.get('http://192.168.1.107:8001/s3/avg_new.pt', timeout=5)
                with open('avg_new.pt', 'wb') as file:
                    file.write(avg_new.content)
            except:
                continue
            break
        time.sleep(5)
    with open('avg.txt', 'wb') as file:
        file.write('-'.encode())
    w_glob_client = torch.load('avg_new.pt')
    
    # Update client-side global model 
    net_glob_client.load_state_dict(w_glob_client)    
    
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
print(loss_train_collect)
print(loss_test_collect)
print(acc_train_collect)
print(acc_test_collect)

#=============================================================================
#                         Program Completed
#=================================================================================