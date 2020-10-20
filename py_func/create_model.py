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
