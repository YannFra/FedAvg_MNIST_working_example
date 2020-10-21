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

"""PLOT THE DISTRIBUTION OF THE TWO CLIENTS"""
def plot_samples(dataset, title=None, plot_name="", n_examples=20):
    
    n_rows = abs(n_examples / 5)
    plt.figure(figsize=(3* n_rows, 3*n_rows))
    if title: plt.suptitle(title)
    for idx,(X,y) in enumerate(dataset.dataset):
        if idx<n_examples:
            ax = plt.subplot(n_rows, 5, idx+1)
            image = 255 - X.view((28,28))
            print(X,y)
            letter = str(y)
            ax.set_title(f"{letter}")
            ax.imshow(image, cmap='gist_gray')
    if plot_name!="":
        plt.savefig(f"plot/"+plot_name+".png")

plot_samples(mnist_train_dls[0],"Client 1")


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
plt.savefig("plots/loss_acc_evolution.png")



# In[2]:  
"""PLOT THE DISTRIBUTION OF A CLIENT"""
def plot_samples(dataset, title=None, plot_name="", n_examples=20):
    
    n_rows = abs(n_examples / 5)
    plt.figure(figsize=(3* n_rows, 3*n_rows))
    if title: plt.suptitle(title)
    for idx,(X,y) in enumerate(dataset.dataset):
        if idx<n_examples:
            ax = plt.subplot(n_rows, 5, idx+1)
            image = 255 - X.view((28,28))
            print(X,y)
            letter = str(y)
            ax.set_title(f"{letter}")
            ax.imshow(image, cmap='gist_gray')
    if plot_name!="":
        plt.savefig(f"plots/"+plot_name+".png")

plot_samples(mnist_train_dls[0],"Client 1", plot_name="samples_client_1")