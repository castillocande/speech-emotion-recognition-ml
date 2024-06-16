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
from glob import glob
import opensmile
import os
from sklearn.model_selection import StratifiedGroupKFold




logger = logging.getLogger(__name__)

class Dataloader():
    def __init__(self):
        pass
        # super(AudioDataloader, self).__init__(*args, **kwargs)
        # self.collate_fn = collate_fn

    def process_dataset(self, data_path, save_path):

        files = glob(data_path)
        actors = np.array([int(path.split('/')[2][-2:]) for path in files])
        print(actors)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        extracted_features = smile.process_files(files)

        x_ = extracted_features.values
        y_ = np.array([int(os.path.basename(path).split('-')[2]) for path in files])

        y_reshaped = y_[:, np.newaxis] 
        dataset = np.concatenate((x_, y_reshaped), axis=1)

        actors = np.array([int(path.split('/')[2][-2:]) for path in files])
        print(actors)
        np.savez(save_path, array1=dataset, array2=actors)
        return np.load(save_path)
    
    def get_dataset(self, path_list):
        dataset = np.load(path_list[0])
        for i in range(1, len(path_list)):
            dataset = np.concatenate((dataset, np.load(path_list[i])))
        x = dataset[:, :-1]
        y = dataset[:,-1]
        return x, y
    

    def split_dataset(self, path_list, by_actors = False, by_distribution = False):
        for i in range(len(path_list)):


            d = StratifiedGroupKFold(n_splits=1)


    


def collate_fn(baches):
    # read preprocessed features or 
    # compute features on-the-fly
    pass
        
