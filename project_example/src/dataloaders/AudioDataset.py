import logging
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data

logger = logging.getLogger(__name__)

class AudioDataset(data.Dataset):
    def __init__(self, metadata):
        pass