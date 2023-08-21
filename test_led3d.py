import argparse
import os
from pathlib import Path
import pdb
import random

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import wandb
import numpy as np

import datasets
import models
import utils
from test import eval_acc
from models.arcface import ArcMarginProduct


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    # val_set = dataset
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    # pdb.set_trace()
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))
        if k == 'inp':
            log(f'  {k}: max:{v.max()} min:{v.min()}')
    # log(dataset.path_print)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return val_loader

    
def prepare_testing():
    sv_file = torch.load(config['ckp'])
    model = models.make(sv_file['model'], load_sd=True).cuda()

    return model


def main(config_, save_path):
    global config, log
    config = config_
    log = utils.set_save_path_(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    log_info = []

    val_loader = make_data_loaders()
    gallery = datasets.make(config.get('gallery'))
    model = prepare_testing()
    is_lock3dface = config.get('is_lock3dface', False)
    total_acc, subset_acc = eval_acc(val_loader, model, gallery,
                                     is_lock3dface=is_lock3dface)
    log_info.append('val: total_acc={:.4f}'.format(total_acc.item()))
    log(', '.join(log_info))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    # parser.add_argument('--name', default=None)
    # parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = config['train_name']
    save_path = os.path.join('./save/led3d-experiment', save_name)

    main(config, save_path)