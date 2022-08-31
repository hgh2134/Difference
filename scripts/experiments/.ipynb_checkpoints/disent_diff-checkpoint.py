import sys, os
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer

# Load experiment ingredients and their respective configs.
from ingredients.dataset import dataset, get_dataset, init_loader, get_data_spliters, group_data
from ingredients.autoencoders import model, init_plgm
from ingredients.training import training, ModelCheckpoint, init_metrics, \
                                 init_optimizer
# from ingredients.analysis import compute_disentanglement

import configs.training as train_params
import configs.cnnvae as model_params

if '../src' not in sys.path:
    sys.path.append('../src')

from training.handlers import Tracer

# Set up experiment
ex = Experiment(name='disent', ingredients=[dataset, model, training])

# Observers
# ex.observers.append(FileStorageObserver.create('../data/sims/disent'))
#ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='disent'))

# General configs
ex.add_config(no_cuda=False, save_folder='../data/sims/dsprites')
#ex.add_package_dependency('torch', torch.__version__)


# Functions

@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device

def input_size_to_paired_input_size(input_size):
    if isinstance(input_size, (list,tuple)):
        return (input_size[0]*2, *input_size[1:])
    elif isinstance(input_size, int):
        return input_size * 2
    else:
        raise
        

# Dataset configs
dataset.add_config(dataset='sprites', setting='unsupervised', shuffle=True)

dataset.add_named_config('dsprites', dataset='dsprites')
dataset.add_named_config('shapes3d', dataset='shapes3d')
dataset.add_named_config('mpi3d', dataset='mpi3d', version='real')

# Training configs
training.config(train_params.vae)
training.named_config(train_params.beta)
training.named_config(train_params.cci)
training.named_config(train_params.factor)
training.named_config(train_params.bsched)
training.named_config(train_params.banneal)


# Model configs
model.named_config(model_params.higgins)
model.named_config(model_params.burgess)
model.named_config(model_params.burgess_v2)
model.named_config(model_params.mpcnn)
model.named_config(model_params.mathieu)
model.named_config(model_params.kim)
model.named_config(model_params.diff_base)


@ex.automain
def main(_config, save_folder):
    print(_config)
    batch_size = _config['training']['batch_size']
    epochs = _config['training']['epochs']
    input_size = _config['model']['input_size']
    paired_input_size = input_size_to_paired_input_size(input_size) 
    flatten = not isinstance(input_size, (list, tuple)) 
    
    device = set_seed_and_device()

    # Load data
    init_dataset = get_dataset()
    data_filters = get_data_spliters()

    dataset = init_dataset(data_filters=data_filters, train=True, flatten = flatten)
    training_dataloader = init_loader(dataset, batch_size)#, device = device


    model = init_plgm(input_size = paired_input_size).to(device=device)
    loss, metrics = init_metrics()
    optimizer = init_optimizer(params=model.parameters())
        
    print('model', model)
    for data, label in training_dataloader:
        data = data.to(device)
        d0, d1, d2, d3, d4 = group_data(data)
        y = model.encoder(d0)
        print('y.shape', y.shape)
        #x1 = model.decoder(y)
        break
        
    trainer = Engine(train_step)
    evaluator = Engine(validation_step)
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    validator = create_supervised_evaluator(model, metrics, device=device)
    
    trainer.run(training_dataloader, max_epochs=epochs)
    
    
def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()



def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y
