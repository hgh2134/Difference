import os
import sys
from io import BytesIO
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import json

from ingredients.autoencoders import init_lgm
from ingredients.decoders import init_decoder
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import configs.datasplits as splits

if '../src' not in sys.path:
    sys.path.append('../src')

import dataset.shapes3d as shapes3d
import dataset.sprites as dsprites
import dataset.mpi as mpi3d

def load_decoder(experiment_path, id, dataset, device):
    path = experiment_path + str(id) +'/'

    meta = path + 'config.json'
    param_vals = path + 'trained-model.pt'

    with open(meta) as f:
        architecture = json.load(f)['model']

    decoder = init_decoder(**architecture, img_size=dataset.img_size,
                           latent_size=dataset.n_factors)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    decoder.load_state_dict(state_dict)

    return decoder.to(device=device).eval()


def load_lgm(experiment_path, id, device):
    path = experiment_path + str(id) + '/'

    meta = path + 'config.json'
    param_vals = path + 'trained-model.pt'

    with open(meta) as f:
        architecture = json.load(f)['model']

    lgm = init_lgm(**architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    lgm.load_state_dict(state_dict)

    return lgm.to(device=device).eval()


#################################### Load Data ################################

@dataclass
class DatasetWrapper:
    raw: tuple
    unsupervised_ctr: type
    supervised_ctr: type
    reconstruction_ctr: type

    def get_unsupervised(self):
        return self.unsupervised_ctr(*self.raw)

    def get_supervised(self):
        return self.supervised_ctr(*self.raw)

    def get_reconstruction(self):
        return self.reconstruction_ctr(*self.raw)

    @property
    def factors(self):
        return self.unsupervised_ctr.lat_names

    @property
    def n_factors(self):
        return self.unsupervised_ctr.n_gen_factors

    @property
    def img_size(self):
        return self.unsupervised_ctr.img_size


def partition_data(raw, mask):
    if len(raw) == 3:
        imgs, latents, latent_classes = raw
        idx = mask(latents, latent_classes)

        imgs = imgs[idx]
        latents = latents[idx]
        latent_classes = latent_classes[idx]

        return imgs, latents, latent_classes

    else:
        imgs, latents = raw
        idx = mask(latents)

        imgs = imgs[idx]
        latents = latents[idx]

        return imgs, latents


def load_dataset(dataset, condition, variant):
    if dataset == 'shapes3d':
        dataset_path = '../data/raw/shapes3d/3dshapes.h5'
        partition_masks = splits.Shapes3D.get_splits(condition, variant)
        dataset_module = shapes3d

    elif dataset == 'dsprites':
        dataset_path = '../data/raw/dsprites/dsprite_train.npz'
        partition_masks = splits.Dsprites.get_splits(condition, variant)
        dataset_module = dsprites

    elif dataset == 'mpi3d':
        dataset_path = '../data/raw/mpi/mpi3d_real.npz'
        partition_masks = splits.MPI3D.get_splits(condition, variant)
        dataset_module = mpi3d

    else:
        raise ValueError('Unrecognized dataset {}'.format(dataset))

    raw = dataset_module.load_raw(dataset_path)
    train_filter, test_filter = partition_masks
    raw_train = partition_data(raw, train_filter)
    raw_test = partition_data(raw, test_filter)

    loaders = (dataset_module.Unsupervised,
               dataset_module.Supervised,
               dataset_module.Reconstruction)

    train_wrapper = DatasetWrapper(raw_train, *loaders)
    test_wrapper = DatasetWrapper(raw_test, *loaders)

    data = train_wrapper, test_wrapper

    return data

def init_loader(dataset, batch_size, **loader_kwargs): #
    kwargs = {'shuffle': True, 'pin_memory': False, 'prefetch_factor': 2,
              'num_workers': 0, 'persistent_workers': False}
    kwargs.update(**loader_kwargs)

    kwargs['pin_memory'] = kwargs['pin_memory'] and torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size, **kwargs)
    #loader = DataLoader(dataset, batch_size, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), **kwargs)

    return loader

def make_pairing(data):    
    '''
        data: 2N X C X H X W
        return: N X 2C X H X W
    '''
    batch_size = data.shape[0]
    assert batch_size%2 == 0
    return data.reshape(batch_size//2, -1, *data.shape[2:])
    
def group_data(data):
    '''
        Make data for check axiom of group
        data_zero: g, g
        data_inverse1:, g, h
        data_inverse2: h, g
        data_assoc1: g,h
        data_assoc2: h, r
    '''
    batch_size = data.shape[0]
    
    index_zero = torch.repeat_interleave(torch.arange(batch_size),2)
    data_zero = make_pairing(data[index_zero])
    
    num_inverse = batch_size//2
    index_inverse1 = torch.arange(batch_size)
    index_inverse2 = torch.arange(batch_size) + torch.tensor([1,-1]).repeat(num_inverse)
    data_inverse1 = make_pairing(data[index_inverse1])
    data_inverse2 = make_pairing(data[index_inverse2])
    
    num_assoc = batch_size//3
    random_batch_index = torch.randperm(num_assoc*3)
    shuffled_data = data[random_batch_index]
    
    index_assoc1= torch.sort(torch.cat((torch.arange(num_assoc)*3, torch.arange(num_assoc)*3+1) ))[0]
    index_assoc2 = torch.sort(torch.cat((torch.arange(num_assoc)*3+1, torch.arange(num_assoc)*3+2) ))[0]
    index_assoc3 = torch.sort(torch.cat((torch.arange(num_assoc)*3, torch.arange(num_assoc)*3+2) ))[0]
    data_assoc1 = make_pairing(shuffled_data[index_assoc1])
    data_assoc2 = make_pairing(shuffled_data[index_assoc2])
    data_assoc3 = make_pairing(shuffled_data[index_assoc3])
    return data_zero, data_inverse1, data_inverse2, data_assoc1, data_assoc2, data_assoc3

#============================== Save Data =====================================

def safe_save(data, path):
    if os.path.exists(path):
        old_data = np.load(path)
        for k in old_data:
            if k not in data:
                data[k] = old_data[k]

    np.savez(path, **data)
