import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from globalvar import *

HIDDEN_LAYER = 50
#Class definitions for NN model and learning algorithm
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, HIDDEN_LAYER)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        
        self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        
        self.fc3 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc3.weight.data.normal_(0, 0.1)   # initialization
        
        self.fc4 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc4.weight.data.normal_(0, 0.1)   # initialization
        
        self.out = nn.Linear(HIDDEN_LAYER, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value