#%% config

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default = 'dsprites', help='Dataset')
parser.add_argument('--condition', type=str, default = 'recomb2element', help='Specify Partitioning Training and Validation dataset')
parser.add_argument('--beta', type = float, default=1, help='Coefficient of Group Loss')
parser.add_argument('--save_folder', type = str, default='../data/sims/test', help='Folder to Save')

args = parser.parse_args()

import sys
class _config:
    encoder_name = 'burgess' #['higgins', 'burgess', 'burgess_v2', 'mpcnn', 'mathieu', 'kim']
    decoder_name = 'CycleGAN'
    latent_size = 10

    dataset_type = args.dataset
    if args.dataset == 'dsprites':
        input_size = [1,64,64]
    elif args.dataset == 'shapes3d':
        input_size = [3,64,64]
    elif args.dataset == 'mpi3d':
        input_size = [3,64,64]
    else:
        raise
    #dataset_type = 'dsprites' # ['dsprites', 'shapes3d', 'mpi3d']
    #condvar = ['recomb2element', 'leave1out']

    if args.condition == "test":
        condvar = ['test', 0.001]
    elif args.condition == "entire":
        #condvar = ['entire', 0.001]
        condvar = ['entire', 0.9999]
    elif args.condition == "recomb2element":
        condvar = ['recomb2element', 'leave1out']
    elif args.condition == "recomb2range":
        condvar = ['recomb2range', 'shape2tx']
    elif args.condition == "extrp":
        condvar = ['extrp', 'blank_side']
    else:
        raise

    '''
        dsprites_combination = [['recomb2range', 'shape2tx'],
        ['recomb2element', 'leave1out'],
        ['extrp', 'blank_side']]

        shaped3d_combination = [['recomb2range', 'shape2ohue'],
        ['recomb2element', 'leave1out'],
        ['extrp', 'fhue_gt50']]

        mpi3d_combination = [['recomb2range', 'cyl2horz'],
        ['recomb2element', 'leave1out'],
        ['extrp', 'horz_gt20']] 
    '''
    batch_size = 64
    learning_rate = 5e-4
    epochs = 100
    #beta = 1
    beta = args.beta

    optimizer = 'adam'
    l2_norm = 0.00

    metrics_config = [{'name': 'recons_nll', 'params': {'loss': 'bce'}, 'output': ['y_pred', 'y']}]
    group_loss = 'l1'
    image_loss = 'bce'
    #save_folder='../data/sims/leave1out_beta100'
    #save_folder= sys.argv[1]
    save_folder= args.save_folder

config = _config()


#%% 
import sys
import torch

if '../src' not in sys.path:
    sys.path.append('../src')
if '../scripts/' not in sys.path:
    sys.path.append('../scripts/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from configs.config import config_sl
config_structure = config_sl(_config)
config_structure.save(config.save_folder)
#%% Define model, dataloader, criterion, optimizer
from ingredients.autoencoders import init_plgm
from ingredients.training import ModelCheckpoint
from analysis.utils import init_loader,  ptr_to_datas,index_to_group_input, ShuffledDataset
from training.loss import latent_diff_loss, image_diff_loss, get_metrics_list, ReconstructionNLL
from training.optimizer import init_optimizer,init_lr_scheduler
from torch import optim
from analysis.utils import load_dataset_pivot
from training.handlers import Saver_concat
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter(config.save_folder)

model = init_plgm(config.encoder_name, config.decoder_name, config.input_size, config.latent_size)
condition, variant = config.condvar
train_data, test_data, pivot_data = load_dataset_pivot(config.dataset_type, condition, variant)
train_dataset = train_data.get_supervised()
test_dataset = test_data.get_supervised()
pivot_dataset = pivot_data.get_supervised()
pairing_dataset = ShuffledDataset(train_dataset, test_dataset)
pivot_pairing_dataset = ShuffledDataset(pivot_dataset, test_dataset)
print('train_dataset', len(train_dataset))
print('test_dataset', len(test_dataset))

train_dataloader = init_loader(train_dataset, batch_size = config.batch_size)
valid_dataloader = init_loader(test_dataset, batch_size = config.batch_size)
pairing_dataloader = init_loader(pairing_dataset, batch_size = config.batch_size)
pivot_pairing_dataloader = init_loader(pivot_pairing_dataset, batch_size = config.batch_size)

criterion_latent = latent_diff_loss(config.group_loss)
criterion_image = image_diff_loss(config.image_loss)

optimizer = init_optimizer(config.optimizer, model.parameters(), lr=config.learning_rate, l2_norm= config.l2_norm)
#optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

metrics = get_metrics_list(config.metrics_config)
metrics_train = get_metrics_list(config.metrics_config)
metrics_pairing = get_metrics_list(config.metrics_config)
metrics_pivot =  get_metrics_list(config.metrics_config)

params = {'loss': 'bce'}
nll_criterion = ReconstructionNLL(**params)

#%% Engine
from ignite.engine import Events, Engine
def update_model(engine, batch):
    imgs, _ = batch
    model.to(device)
    imgs = imgs.to(device)

    batch_concat_data_conat, batch_concat_data1, batch_concat_data2, ptrs = index_to_group_input(imgs)   
    z = model.encoder(batch_concat_data_conat)
    recon = model.decode(z, batch_concat_data1)
    loss_recon = criterion_image(recon, batch_concat_data2)
    loss_nll = nll_criterion(recon, batch_concat_data2)
    z_iden, z_inv_gh, z_inv_hg, z_assoc_gh, z_assoc_hk, z_assoc_gk = ptr_to_datas(z, ptrs)

    loss_iden = criterion_latent(z_iden, torch.zeros_like(z_iden))
    loss_inv = criterion_latent(z_inv_gh + z_inv_hg, torch.zeros_like(z_inv_gh))
    loss_assoc = criterion_latent(z_assoc_gh + z_assoc_hk, z_assoc_gk)
    
    loss_group = loss_iden + loss_inv + loss_assoc
    loss = loss_recon + config.beta * loss_group
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'y_pred': recon, 
            'y': batch_concat_data2,
            'template': batch_concat_data1,
            'z': z,
            'ptrs': ptrs,
            'loss': loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_iden': loss_iden.item(),
            'loss_inv': loss_inv.item(),
            'loss_assoc': loss_assoc.item(),
            'loss_group': loss_group.item(),
            'loss_nll': loss_nll.item()}
    
def evalutate_model(engine, batch):
    model.eval()
    imgs, _ = batch
    model.to(device)
    imgs = imgs.to(device)

    with torch.no_grad():
        batch_concat_data_conat, batch_concat_data1, batch_concat_data2, ptrs = index_to_group_input(imgs)
        z = model.encoder(batch_concat_data_conat)
        recon = model.decode(z, batch_concat_data1)
        loss_recon = criterion_image(recon, batch_concat_data2)
        loss_nll = nll_criterion(recon, batch_concat_data2)
        z_iden, z_inv_gh, z_inv_hg, z_assoc_gh, z_assoc_hk, z_assoc_gk = ptr_to_datas(z, ptrs)

        loss_iden = criterion_latent(z_iden, torch.zeros_like(z_iden))
        loss_inv = criterion_latent(z_inv_gh + z_inv_hg, torch.zeros_like(z_inv_gh))
        loss_assoc = criterion_latent(z_assoc_gh + z_assoc_hk, z_assoc_gk)
        loss_group = loss_iden + loss_inv + loss_assoc
        loss = loss_recon + config.beta * loss_group
    return {'y_pred': recon, 
            'y': batch_concat_data2,
            'template': batch_concat_data1,
            'z': z,
            'ptrs': ptrs,
            'loss': loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_iden': loss_iden.item(),
            'loss_inv': loss_inv.item(),
            'loss_assoc': loss_assoc.item(),
            'loss_group': loss_group.item(),
            'loss_nll': loss_nll.item()}

def evalutate_model_pairing(engine, batch):
    model.eval()
    imgs_train, _, imgs_test, _ = batch
    model.to(device)
    imgs_train = imgs_train.to(device)
    imgs_test = imgs_test.to(device)
    imgs_concat = torch.cat([imgs_train, imgs_test], dim = 1)

    with torch.no_grad():
        z = model.encoder(imgs_concat)
        recon = model.decode(z, imgs_train)
        loss_recon = criterion_image(recon, imgs_test)
        loss_nll = nll_criterion(recon, imgs_test)

    return {'y_pred': recon, 
            'y': imgs_test,
            'template': imgs_train,
            'z': z,
            'loss_recon': loss_recon.item(),
            'loss_nll': loss_nll.item()}

trainer = Engine(update_model)
validator = Engine(evalutate_model)
validator_pairing = Engine(evalutate_model_pairing)
validator_pivot = Engine(evalutate_model_pairing)

#%% Handlers
for name, metric in metrics.items():
    metric.attach(validator, name)
for name, metric in metrics_train.items():
    metric.attach(trainer, name)   
for name, metric in metrics_pairing.items():
    metric.attach(validator_pairing, name)   
for name, metric in metrics_pivot.items():
    metric.attach(validator_pivot, name)   


output_list = ['loss','loss_recon', 'loss_iden','loss_inv', 'loss_assoc', 'loss_group', 'loss_nll']

metric_saver_train = Saver_concat(save_folder = config.save_folder, output = output_list, name ='train').attach(trainer)
metric_saver_validation = Saver_concat(save_folder = config.save_folder, output = output_list, name ='valid').attach(validator)
metric_saver_pairing = Saver_concat(save_folder = config.save_folder, output = ['loss_recon', 'loss_nll'], name ='pairing').attach(validator_pairing)
metric_saver_pivot = Saver_concat(save_folder = config.save_folder, output = ['loss_recon', 'loss_nll'], name ='pivot').attach(validator_pivot)



@trainer.on(Events.EPOCH_COMPLETED)
def validate(engine):
    validator.run(valid_dataloader)
    validator_pairing.run(pairing_dataloader)
    validator_pivot.run(pivot_pairing_dataloader)



# Attach model checkpoint
def score_fn(engine):
    return -engine.state.metrics[list(metrics)[0]]

best_checkpoint = ModelCheckpoint(
    dirname=config.save_folder,
    filename_prefix='diff_best_nll',
    score_function=score_fn,
    create_dir=True,
    require_empty=False,
    save_as_state_dict=True
)
validator.add_event_handler(Events.COMPLETED, best_checkpoint,
                            {'model': model})

# Save every 10 epochs
periodic_checkpoint = ModelCheckpoint(
    dirname= config.save_folder,
    filename_prefix='diff_interval',
    n_saved=config.epochs//10,
    create_dir=True,
    require_empty=False,
    save_as_state_dict=True
)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10),
                            periodic_checkpoint, {'model': model})

#%% Run
trainer.run(train_dataloader, max_epochs = config.epochs)