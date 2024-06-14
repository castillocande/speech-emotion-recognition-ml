import logging
import importlib
import os, random
import torch
from os.path import join, split, exists

from src.dataloaders.AudioDataloader import AudioDataloader
from src.dataloaders.AudioDataset import AudioDataset

from src.utils.files import create_result_folder
from src.callbacks.EarlyStopper import EarlyStopper
from src.datasets.load_metadata import load_metadata
from src.features.extract_features import extract_features

from src.train_and_test.train import train
from src.train_and_test.test import test

logger = logging.getLogger(__name__)


def run_experiment(model_params, data_params, features_params):
    """
    Este ejemplo es para un caso en el cual solamente entrenemos y evaluemos un modelo
    basado en redes. La idea sera modificar estos pasos para que se ajusten a las necesidades
    de lo que cada uno vaya a proponer como modelo. Podria mantenerse esto mismo y simplemente
    complejizar las subfunciones, o bien modificar esta funcion para que se ajuste a un flujo de
    trabajo distinto, eliminando o agregando pasos.
    """
    context = {}
    # create result folder
    context['save_path'] = create_result_folder(model_params, data_params, features_params)
    
    # get metadata
    metadata = load_metadata(data_params)
    metadata.to_pickle(os.path.join(context['save_path'],'metadata.pkl'))
    
    # extract features
    features = extract_features(metadata, features_params)
    
    # dataloaders
    train_loader = get_dataloader(features, model_params.dataloader_params, 'Train')
    val_loader = get_dataloader(features, model_params.dataloader_params, 'Val')
    test_loader = get_dataloader(features, {}, 'Test')
    
    # load model
    model_params.input_dim = features_params.dim
    model = load_model(model_params)
    
    # train
    early_stopper = EarlyStopper(patience=model_params.early_stop_patience)
    model = train(model, train_loader, val_loader, early_stopper, model_params, context)
    
    # test
    test(test_loader, context)
    logger.info('Experiment completed')
    return 

def load_model(params):
    model_module = importlib.import_module(f'src.models.{params.name}')
    model = getattr(model_module, params.name)(params)
    model = model.to(params.device)
    return model

def get_dataloader(features, params, split):
    features = features[features.split==split]
    dataset = AudioDataset(features)
    dataloader = AudioDataloader(dataset, **params)
    return dataloader

