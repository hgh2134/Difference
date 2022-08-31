import os
import sys
from io import BytesIO
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import json
import dill

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

def load_plgm(experiment_path, id, device):
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

def load_dataset_pivot(dataset, condition, variant):
    if dataset == 'shapes3d':
        dataset_path = '../data/raw/shapes3d/3dshapes.h5'
        partition_masks = splits.Shapes3D.get_splits(condition, variant)
        dataset_module = shapes3d
        pivot_masks, _ = splits.Shapes3D.get_splits('pivot', None)
    elif dataset == 'dsprites':
        dataset_path = '../data/raw/dsprites/dsprite_train.npz'
        partition_masks = splits.Dsprites.get_splits(condition, variant)
        dataset_module = dsprites
        pivot_masks, _ = splits.Dsprites.get_splits('pivot', None)
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
    raw_pivot = partition_data(raw, pivot_masks)

    loaders = (dataset_module.Unsupervised,
                dataset_module.Supervised,
                dataset_module.Reconstruction)

    train_wrapper = DatasetWrapper(raw_train, *loaders)
    test_wrapper = DatasetWrapper(raw_test, *loaders)
    pivot_wrapper = DatasetWrapper(raw_pivot, *loaders)
    data = train_wrapper, test_wrapper, pivot_wrapper

    return data

import random
class ShuffledDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super(ShuffledDataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.random_index = random.choices(range(len(self.dataset1)), k = len(self.dataset2))
    def __len__(self):
        return len(self.dataset2) 

    def __getitem__(self, index):
        data2 = self.dataset2[index]
        data1 = self.dataset1[self.random_index[index]]
        if not (type(data2) == list or type(data2) == tuple):
            data2 = [data2]
        if not (type(data1) == list or type(data1) == tuple):
            data1 = [data1]
        return data1 + data2

def init_loader(dataset, batch_size, **loader_kwargs): #
    kwargs = {'shuffle': True, 'pin_memory': False, 'prefetch_factor': 2,
              'num_workers': 0, 'persistent_workers': False}
    kwargs.update(**loader_kwargs)

    kwargs['pin_memory'] = kwargs['pin_memory'] and torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size, **kwargs)
    #loader = DataLoader(dataset, batch_size, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), **kwargs)

    return loader



def identity_index(data):
    batch_size = data.shape[0]
    index_identity_g = torch.arange(batch_size)
    #data_identity_g = data[index_identity_g]

    #data_itentity = torch.cat([data_identity_g, data_identity_g], dim = 1)
    #return [data_itentity, data_identity_g, data_identity_g], [index_identity_g, index_identity_g]
    return index_identity_g

def inverse_index(data):
    batch_size = data.shape[0]
    num_inverse = batch_size//2
    index_inverse_g = torch.arange(num_inverse)*2 
    index_inverse_h = index_inverse_g + 1

    #data_inverse_g = data[index_inverse_g]
    #data_inverse_h = data[index_inverse_h]

    #data_inverse_gh = torch.cat([data_inverse_g, data_inverse_h], dim = 1)
    #data_inverse_hg = torch.cat([data_inverse_h, data_inverse_g], dim = 1)
    #return [data_inverse_gh, data_inverse_g, data_inverse_h], [data_inverse_hg, data_inverse_h, data_inverse_g], [index_inverse_g, index_inverse_h]
    return index_inverse_g, index_inverse_h

def associativity_index(data):
    batch_size = data.shape[0]
    num_assoc = batch_size//3
    random_batch_index = torch.randperm(num_assoc*3)

    index_assoc_g = torch.arange(num_assoc)*3
    index_assoc_h = index_assoc_g + 1
    index_assoc_r = index_assoc_g + 2
    shuffled_index_g = random_batch_index[index_assoc_g]
    shuffled_index_h = random_batch_index[index_assoc_h]
    shuffled_index_r = random_batch_index[index_assoc_r]

    #data_assoc_g = data[shuffled_index_g]
    #data_assoc_h = data[shuffled_index_h]
    #data_assoc_r = data[shuffled_index_r]

    #data_assoc_gh = torch.cat([data_assoc_g, data_assoc_h], dim = 1)
    #data_assoc_hr = torch.cat([data_assoc_h, data_assoc_r], dim = 1)
    #data_assoc_gr = torch.cat([data_assoc_g, data_assoc_r], dim = 1)

    #return [data_assoc_gh, data_assoc_g, data_assoc_h], [data_assoc_hr, data_assoc_h, data_assoc_r], [data_assoc_gr, data_assoc_g, data_assoc_r], [shuffled_index_g, shuffled_index_h, shuffled_index_r]
    return shuffled_index_g, shuffled_index_h, shuffled_index_r

def index_to_group_input(imgs):

    ind_iden, ind_inv1, ind_inv2, ind_assoc1, ind_assoc2, ind_assoc3 = group_index(imgs)
    list_data_concat = []
    list_data1 = []
    list_data2 = []
    
    for index_combination in [[ind_iden, ind_iden], [ind_inv1, ind_inv2], [ind_inv2, ind_inv1], [ind_assoc1, ind_assoc2], [ind_assoc2, ind_assoc3], [ind_assoc1, ind_assoc3]]:
        index1, index2 = index_combination
        data_concat, data1, data2 = make_pairing(imgs, index1, index2)
        list_data_concat.append(data_concat)
        list_data1.append(data1)
        list_data2.append(data2)
    ptrs = datas_to_ptr(list_data_concat)  
    batch_concat_data_conat = torch.cat(list_data_concat, dim = 0)
    batch_concat_data1 = torch.cat(list_data1, dim = 0)
    batch_concat_data2 = torch.cat(list_data2, dim = 0)
    return batch_concat_data_conat, batch_concat_data1, batch_concat_data2, ptrs

def group_index(data):
    '''
        Make data for check axiom of group
        data_zero: g, g
        data_inverse1:, g, h
        data_inverse2: h, g
        data_assoc1: g, h
        data_assoc2: h, r
        data_assoc3: g, r
    '''

    index_identity_g = identity_index(data)
    index_inverse_g, index_inverse_h = inverse_index(data)
    index_assoc_g, shuffled_index_h, index_assoc_r = associativity_index(data)

    #data_dict = { 'data_identity': data_identity, 
    #'data_inverse1': data_inverse1, 'data_inverse2':data_inverse2,
    #'data_assoc1': data_assoc1, 'data_assoc2': data_assoc2, 'data_assoc3':data_assoc3
    #}
    #index_dict = {'index_identity':index_identity,  'index_inverse': index_inverse, 'index_assoc':index_assoc}
    return index_identity_g, index_inverse_g, index_inverse_h, index_assoc_g, shuffled_index_h, index_assoc_r

def make_pairing(data, index1, index2):
    '''
        data: N x C x H x W
        index1: M
        index2: M
        return: M x 2C x H x W, M x C x H x W, M x C x H x W
    '''
    data1 = data[index1]
    data2 = data[index2]
    data_cat = torch.cat([data1, data2], dim = 1)
    return data_cat, data1, data2

def datas_to_ptr(datas):
    ptrs = []
    idx = 0
    ptrs.append(idx)
    for data in datas:
        length = data.shape[0]
        idx += length
        ptrs.append(idx)
    return ptrs

def ptr_to_datas(data, ptrs):
    assert ptrs[-1] == data.shape[0]
    n = len(ptrs) - 1
    datas = []
    for i in range(n):
        datas.append(data[ptrs[i]:ptrs[i+1]])
    return datas
#============================== Save Data =====================================

def safe_save(data, path):
    if os.path.exists(path):
        old_data = np.load(path)
        for k in old_data:
            if k not in data:
                data[k] = old_data[k]

    np.savez(path, **data)
