#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
    

class NN(nn.Module):
    
    def __init__(self,layer_1,layer_2):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784,layer_1)
        self.fc3 = nn.Linear(layer_1,10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1,784)))
        x=self.fc3(x)
        return x
    
    
class CNN(nn.Module):

   """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
   def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

   def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
