import argparse
import os
import math
from functools import partial
import pdb
import math

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pytorch_msssim import ssim
from pathlib import Path

import datasets
import models
import utils

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        if inp.shape[1] == 3 or inp.shape[1] == 1:
            model.gen_feat(inp)
        elif inp.shape[1] == 6:
            inp_depth = inp[:, :3, :, :]
            inp_normal = inp[:, 3:, :, :]
            model.gen_feat(inp_depth, inp_normal)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


def eval_acc(loader, model, gallery, data_norm=None, is_lock3dface=True,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()


    total_acc = utils.Accuracy()
    subset_acc = {
        'NU': utils.Accuracy(),
        'FE': utils.Accuracy(),
        'PS': utils.Accuracy(),
        'OC': utils.Accuracy(),
        'TM': utils.Accuracy(),
    }
    gallery_feats, gallery_labels = \
        utils.extract_gallery_features(model, gallery)

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, list):
                batch[k] = [e.cuda() for e in v]
            else:
                batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        label = batch['label']
        basic_subsets = batch.get('basic_subset', None)
        TM_subsets = batch.get('TM_subset', None)
        # basic_subsets = batch['basic_subset']
        # TM_subsets = batch['TM_subset']
        with torch.no_grad():
            probe_feats = model(inp, is_train=False)
            norm_probe_feats = F.normalize(probe_feats, p=2, dim=-1)
            norm_gallery_feats = F.normalize(gallery_feats, p=2, dim=-1)
            sim = torch.matmul(norm_probe_feats, norm_gallery_feats.T)
            idx = torch.argmax(sim, dim=1)
            preds = gallery_labels[idx]
            # shape
            total_acc.add(preds, label)
            if is_lock3dface:
                for i, (k, v) in enumerate(subset_acc.items()):
                    if TM_subsets.all():
                        break
                    if k == 'TM':
                        continue
                    if not (basic_subsets == i).any():
                        continue
                    index_tensor = (basic_subsets == i) & (~TM_subsets)
                    if index_tensor.any():
                        v.add(preds[index_tensor], label[index_tensor])
                if TM_subsets.any():
                    subset_acc['TM'].add(preds[TM_subsets], label[TM_subsets])
                    
        if verbose:
            pbar.set_description('val {:.4f}'.format(total_acc.item()))

        torch.cuda.empty_cache()
    return total_acc, subset_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
