import importlib
import torch
import torch.nn as nn 
import numpy as np
import logging
from torch import nn, optim

logger = logging.getLogger(__name__)

class NNDense(nn.Module):
    def __init__(self, params):
        super(Dense, self).__init__()
        pass
        
    def forward(self, x):
        pass
    
    def train_step(self, data):
        pass

    def eval_step(self, data):
        pass