#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

# In[2]:
"""CREATING DATASETS FOR DIFFERENT FONTS"""

fonts = ['InconsolataN']#,'jsMath-cmti10']
all_digits=[i for i in range(10)]

C1= {
     'n_samples_train': 200,
     'font':'InconsolataN',
     'numbers': all_digits,
     'tilt':0,
     'std_tilt': 10, #std on the tilt,
     'seed':0
     }
C1['n_samples']= int(1.5 * C1['n_samples_train']) #20% more for the testing set

C2=deepcopy(C1)
C2['tilt']=45

clients = [C1, C2]


# In[2]:
"""UPLOAD THE DATASETS"""
from py_func.create_custom_MNIST import MNIST_custom_train_test_sets
custom_mnist_train, custom_mnist_test = MNIST_custom_train_test_sets(clients)


# In[2]: 
"""PLOT THE DISTRIBUTION OF A CLIENT"""
from py_func.plot_samples_clients import plot_samples
plot_samples(custom_mnist_train[0], "Client 1 custom")
# In[2]: 
"""LOAD THE MODEL"""
from py_func.create_model import CNN
model_0 = CNN()


# In[2]: 
"""RUN FEDAVG""" 
n_iter=10

from py_func.FedProx import FedProx
model_f, loss_hist, acc_hist = FedProx( model_0, custom_mnist_train, 
    n_iter, custom_mnist_test, epochs=5, lr=0.1)

  
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
