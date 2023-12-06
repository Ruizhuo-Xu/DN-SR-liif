import argparse
import os
import math
import pdb

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import wandb

import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import datasets
import models
import utils
from test import eval_psnr

# defines two global scope variables to store our gradients and activations 
gradients = None 
activations = None 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
     torch.backends.cudnn.deterministic = True

def ddp_setup(rank, world_size, port='12355'):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_reduce(x):
    res  = torch.tensor(x).cuda()
    dist.reduce(res, 0)
    return res.item()

def backward_hook(module, grad_input, grad_output): 
    global gradients # refers to the variable in the global scope 
    gradients = grad_output 

def forward_hook(module, args, output): 
    global activations # refers to the variable in the global scope 
    activations = output 

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=spec.get('num_workers', 0),
        pin_memory=True,
        collate_fn=utils.RandomDwonSample(scale_min=spec['scale_min'], scale_max=spec['scale_max']),
        sampler=DistributedSampler(dataset, shuffle=(tag=='train'))
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    if dist.get_rank() == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def get_id_model():
    if config.get('id_net'):
        sv_file = torch.load(config['id_net'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        # 冻结识别网络参数
        for name, parameter in model.named_parameters():
            parameter.requires_grad = False
        if dist.get_rank() == 0:
            log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
        return model
    else:
        return None


def train(train_loader, model, optimizer, id_model=None):
    model.train()
    if config.get('WeightedL1Loss'):
        l1_loss_fn = utils.WeightedL1Loss((128, 128))
    else:
        l1_loss_fn = nn.L1Loss()
    id_loss_fn = utils.CosineSimilarityLoss()
    # id_loss_fn = nn.L1Loss()

    l1_loss_avg = utils.Averager()
    id_loss_avg = utils.Averager()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    with tqdm(train_loader, leave=False, desc='train', ascii=True) as t:
        for iter_step, batch in enumerate(t, 1):
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = (batch['inp'] - inp_sub) / inp_div
            # B, N, C
            pred = model(inp, batch['coord'], batch['cell'])

            gt = (batch['gt'] - gt_sub) / gt_div

            bs, n_coords, channels = gt.shape
            side = int(math.sqrt(n_coords))
            pred = pred.view(bs, side, side, channels).permute(0, 3, 1, 2)
            gt = gt.view(bs, side, side, channels).permute(0, 3, 1, 2)

            optimizer.zero_grad()
            if id_model:
                id_loss_weight = config.get('id_loss_weight', 0)
                _backward_hook = id_model.block5.register_full_backward_hook(backward_hook) 
                _forward_hook = id_model.block5.register_forward_hook(forward_hook)
                id_loss = id_loss_weight * utils.calc_id_loss(pred, gt, id_model, id_loss_fn)
                # 先backward计算L1 Loss权重
                id_loss.backward(retain_graph=True)
                heatmap = utils.grad_cam_heatmap(gradients[0], activations)
                _backward_hook.remove()
                _forward_hook.remove()
            else:
                id_loss = torch.Tensor([0]).cuda()
                heatmap = None

            if config.get('WeightedL1Loss'):
                l1_loss_weight = config.get('l1_loss_weight', 1.0)
                l1_loss = l1_loss_fn(pred, gt, heatmap)
            else:
                l1_loss = l1_loss_fn(pred, gt)
            loss = l1_loss + id_loss

            l1_loss_avg.add(l1_loss.item())
            id_loss_avg.add(id_loss.item())
            train_loss.add(loss.item())

            l1_loss.backward()
            # pdb.set_trace()
            optimizer.step()

            tqdm.set_postfix(t, {'loss': train_loss.item(),
                                 'l1_loss': l1_loss_avg.item(),
                                 'id_loss': id_loss_avg.item()
                                 })

            pred = None; loss = None;

    return l1_loss_avg, id_loss_avg, train_loss


def main(rank, world_size, config_, save_path, args):
    global config, log
    ddp_setup(rank, world_size, args.port)
    config = config_
    if rank == 0:
        save_name = save_path.split('/')[-1]
        wandb.init(project='liif-sr', name=save_name)
        log = utils.set_save_path(save_path)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    dist.barrier()
    torch.cuda.set_device(rank)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    id_model = get_id_model()
    # if id_model:
    #     id_model = DDP(id_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        l1_loss, id_loss, train_loss = train(train_loader, model, optimizer, id_model)
        l1_v = ddp_reduce(l1_loss.v)
        l1_n = ddp_reduce(l1_loss.n)
        id_v = ddp_reduce(id_loss.v)
        id_n = ddp_reduce(id_loss.n)
        train_v = ddp_reduce(train_loss.v)
        train_n = ddp_reduce(train_loss.n)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            l1_loss = l1_v / l1_n
            id_loss = id_v / id_n
            train_loss = train_v / train_n
            log_info.append('train: l1_loss={:.4f}'.format(l1_loss))
            log_info.append('train: id_loss={:.4f}'.format(id_loss))
            log_info.append('train: total_loss={:.4f}'.format(train_loss))
            wandb.log({
                'train/loss': train_loss,
                'train/id_loss': id_loss,
                'train/l1_loss': l1_loss,
                'lr': current_lr
            }, epoch)

        # if n_gpus > 1:
        #     model_ = model.module
        # else:
        #     model_ = model
        model_spec = config['model']
        model_spec['sd'] = model.module.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        if rank == 0:
            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res = eval_psnr(val_loader, model.module,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            v = ddp_reduce(val_res.v)
            n = ddp_reduce(val_res.n)
            if rank == 0:
                val_res = (v / n)
                log_info.append('val: psnr={:.4f}'.format(val_res))
                wandb.log(
                    {'val/psnr': val_res}, epoch
                )
                if val_res > max_val_v:
                    max_val_v = val_res
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        if rank == 0:
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            log(', '.join(log_info))
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', default='12355')
    parser.add_argument('--save_path', default='./save')
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enabling torch.Compile')
    parser.add_argument('--WeightedL1Loss', action='store_true', default=False)
    parser.add_argument('--id_loss_weight', type=float, default=1.0)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join(args.save_path, save_name)

    # main(config, save_path)
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config, save_path, args), nprocs=world_size)
