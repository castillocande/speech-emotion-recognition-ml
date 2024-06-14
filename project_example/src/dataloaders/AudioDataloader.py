import librosa
import logging
import random
import torch
import time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils import data
from src.dataloaders.AudioDataset import AudioDataset

logger = logging.getLogger(__name__)

class AudioDataloader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn

def collate_fn(baches):
    # read preprocessed features or 
    # compute features on-the-fly
    pass
        
