"""
Sacred Ingredient for datasets

This ingredient has the functions to load datsets and DataLoaders for training.
Selecting a dataset is a matter of passing the corresponding name. There is a
function to get the splits, and one to show them (assuming they are iamges).

Three datasets are currently supported, dSprites, 3DShapes and MPI3D. The
transformation dataset can also be loaded using this function.
"""


import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sacred import Ingredient

import matplotlib.pyplot as plt

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

# from dataset.tdisc import load_tdata
from dataset.sprites import load_sprites
from dataset.shapes3d import load_shapes3d
from dataset.mpi import load_mpi3d
from dataset.transforms import Triplets

import configs.datasplits as splits


dataset = Ingredient('dataset')
load_sprites = dataset.capture(load_sprites)
load_shapes3d = dataset.capture(load_shapes3d)
load_mpi3d = dataset.capture(load_mpi3d)
load_composition = dataset.capture(Triplets)

dataset.add_config(setting='unsupervised')
dataset.add_named_config('unsupervised', setting='unsupervised')
dataset.add_named_config('supervised', setting='supervised')


@dataset.capture
def get_dataset(dataset):
    if dataset == 'dsprites':
        dataset_loader = load_sprites
    elif dataset == 'shapes3d':
        dataset_loader = load_shapes3d
    elif dataset == 'mpi3d':
        dataset_loader = load_mpi3d
    elif dataset == 'composition':
        dataset_loader = load_composition
    else:
        raise ValueError('Unrecognized dataset {}'.format(dataset))

    return dataset_loader


@dataset.capture
def init_loader(dataset, batch_size, **loader_kwargs): #
    kwargs = {'shuffle': True, 'pin_memory': False, 'prefetch_factor': 2,
              'num_workers': 0, 'persistent_workers': False}
    kwargs.update(**loader_kwargs)

    kwargs['pin_memory'] = kwargs['pin_memory'] and torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size, **kwargs)
    #loader = DataLoader(dataset, batch_size, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), **kwargs)

    return loader

@dataset.command(unobserved=True)
def plot():
    dataset = get_dataset()(setting='supervised')
    loader = init_loader(dataset, 1, pin_memory=False,
                         shuffle=False, n_workers=1)

    for img, t in loader:
        img = img.reshape(loader.dataset.img_size)
        img = img.squeeze().numpy()

        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
            cmap = None
        else:
            cmap = 'Greys_r'

        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.show(block=True)


@dataset.capture
def get_data_spliters(dataset, condition=None, variant=None):
    if condition is None:
        return None, None
    if dataset == 'dsprites':
        return splits.Dsprites.get_splits(condition, variant)
    elif dataset == 'shapes3d':
        return splits.Shapes3D.get_splits(condition, variant)
    elif dataset == 'mpi3d':
        return splits.MPI3D.get_splits(condition, variant)
    else:
        raise ValueError('Condition given,'
                         'but dataset {} is invalid'.format(dataset))

        
        
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
    data_assoc1 = make_pairing(shuffled_data[index_assoc1])
    data_assoc2 = make_pairing(shuffled_data[index_assoc2])
    return data_zero, data_inverse1, data_inverse2, data_assoc1, data_assoc2