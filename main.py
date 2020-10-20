#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

from copy import deepcopy
import torch.optim as optim

# In[2]:  
"""UPLOADING THE DATASETS"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_trainset=datasets.MNIST(root='./data', train=True, download=True, 
    transform=transforms.ToTensor())
mnist_train_split = torch.utils.data.random_split(mnist_trainset, 
    [200, 200, 200, 60000 -3*200])[:-1]
mnist_train_dls =[torch.utils.data.DataLoader(ds, batch_size=10, 
    shuffle=True) for ds in mnist_train_split]

mnist_testset=datasets.MNIST(root='./data', train=False, download=True, 
    transform=transforms.ToTensor()) 
mnist_test_split = torch.utils.data.random_split(mnist_testset, 
    [100, 100, 100, 10000 -3*100])[:-1]
mnist_test_dls =[torch.utils.data.DataLoader(ds, batch_size=10, 
    shuffle=True) for ds in mnist_test_split]

# In[2]: 
"""LOAD THE MODEL"""
from py_func.create_model import NN
model_0=NN(50,10)


# In[2]: 
"""RUN FEDAVG""" 
n_iter=10

from py_func.FedProx import FedProx
model_f, loss_hist, acc_hist = FedProx( model_0, mnist_train_dls, 
    n_iter, mnist_test_dls)

  
# In[2]:  
"""PLOT THE LOSS AND ACC HISTORY FOR THE DIFFERENT CLIENTS"""
import matplotlib.pyplot as plt

plt.figure()

plt.subplot(1,2,1)
lines=plt.plot(loss_hist)
plt.title("Loss")
plt.legend(lines,["C1", "C2", "C3"])

plt.subplot(1,2,2)
lines=plt.plot(acc_hist, )
plt.title("Accuracy")
plt.legend(lines, ["C1", "C2", "C3"])