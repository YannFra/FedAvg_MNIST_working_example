#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# prerequisites
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
import os

#import matplotlib.pyplot as plt
from copy import deepcopy


# In[2]:
"""CREATING DATASETS FOR DIFFERENT FONTS"""

fonts = ['InconsolataN']#,'jsMath-cmti10']
all_digits=[i for i in range(10)]

C1= {
     'n_samples_train': 20,
     'font':'InconsolataN',
     'numbers': all_digits,
     'tilt':0,
     'std_tilt': 10, #std on the tilt,
     'seed':0
     }
C1['n_samples']= int(1.2 * C1['n_samples_train']) #20% more for the testing set

C2=deepcopy(C1)
C2['tilt']=30

clients = [C1, C2]


# In[2]:
"""CREATION OF THE DATASET FOR EACH CLIENT IF THERE IS NOT ONE"""
from py_func.create_custom_MNIST import build_dataset, save_dataset


for Ci in clients:
    build_dataset(Ci)
     

# In[2]:
"""Creation of the dataset for each client if they do not exist"""
from py_func.create_custom_MNIST import build_dataset, save_dataset


for family in families:
    
    file_name=family +" "+ numbers_str+" "+str(training_samples)
    plot_name=family +" "+ numbers_str
    
    if tilted_angle!=0:
        file_name+="_"+str(tilted_angle)
        plot_name+="_"+str(tilted_angle)
    
    if not os.path.isfile('dataset/'+file_name+".pkl"):
        
        print(family)
    
        feature_data, label_data= build_dataset(n_samples,family,std_rotation,
            std_size,plot_name,dataset_digits=numbers,tilted_angle=tilted_angle)
#        save_dataset(file_name, feature_data,label_data,tilted_angle=tilted_angle)
     

# In[2]:
import random
np.random.seed(0)
np.random.randint(10)
random.choice([i for i in range(10)])

