import logging 
import torch
import time
import numpy as np
from src.callbacks.Writer import TensorBoardWriter
from os.path import join, split, sep
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score

logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, early_stopper, params, context):
    if params.tboard:
        writer = TensorBoardWriter(context['save_path'])
    prev_val_loss = float("inf")
    best_val_loss = float("inf")
    
    for epoch in range(1, params.max_epochs+1):
        model.train()
        # train loop
        # ...
        
        model.eval()
        # validation loop
        # ...
        
        # callbacks calls
        # ...

        writer.update_status(epoch, train_loss, val_loss)
        if (val_loss+0.001) < best_val_loss:
            pass
        if epoch == params.max_epochs:
            pass
        if not _continue:
            pass
        

