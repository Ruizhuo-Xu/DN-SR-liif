import os
import time
import shutil
import math
import random

import torch
from torch import nn
import numpy as np
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import transforms
from PIL import Image


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Accuracy():

    def __init__(self):
        self.correct_num = 0
        self.total_num = 0
        self.acc = 0.0

    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        assert preds.shape[0] == labels.shape[0]
        correct_num = (preds == labels).sum().item()
        total_num = preds.shape[0]
        self.correct_num += correct_num
        self.total_num += total_num
        self.acc = self.correct_num / self.total_num

    def item(self):
        return self.acc


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def set_save_path_(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    # writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(img.shape[0], -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def add_noise_Guass(img, mean=0, var=0.01):  # 添加高斯噪声
    img_copy = img.copy()
    img = (img / 255).astype(np.float64)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out_img = img + noise
    out_img = np.clip(out_img, 0.0, 1.0)
    out_img[img_copy==0] = 0
    out_img = out_img * 255
    return out_img.astype(np.uint8)


def calc_normalMap(depth_map):
    """calculate normal map from depth map

    Args:
        depth_map (ndarray): depth map

    Returns:
        ndarray: normal map
    """
    d_im = depth_map.astype("float64")
    zy, zx = np.gradient(d_im)  
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    #zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
    #zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    # if show, comment
    normal *= 255
    # cv2.imwrite("normal.png", normal[:, :, ::-1])
    return normal[:,:,::-1].copy()


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


def random_downsample(scale_min=1.0, scale_max=4.0):
    assert scale_min > 0 and scale_max > 0 and scale_max >= scale_min
    def collect_fn(batch):
        noisy_imgs, labels, basic_subsets, TM_subsets = [], [], [], []
        hr_coords, hr_gts, cells = [], [], []
        # 同一个batch进行相同的降采样
        s = random.uniform(scale_min, scale_max)
        for data in batch:
            img_noisy, img_ori, label, basic_subset, TM_subset = data
            labels.append(label)
            basic_subsets.append(basic_subset)
            TM_subsets.append(TM_subset)
            # downsample
            w_hr = img_ori.shape[-1]
            w_lr = round(w_hr / s)
            img_noisy_lr = resize_fn(img_noisy, w_lr)
            hr_coord, hr_gt = to_pixel_samples(img_ori)
            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / w_hr
            cell[:, 1] *= 2 / w_hr
            hr_coords.append(hr_coord)
            hr_gts.append(hr_gt)
            cells.append(cell)

            noisy_imgs.append(img_noisy_lr)
        noisy_imgs = torch.stack(noisy_imgs, dim=0)
        hr_coords = torch.stack(hr_coords, dim=0)
        hr_gts = torch.stack(hr_gts, dim=0)
        cells = torch.stack(cells, dim=0)
        labels = torch.stack(labels, dim=0)
        basic_subsets = torch.stack(basic_subsets, dim=0)
        TM_subsets = torch.stack(TM_subsets, dim=0)
        return {
            'inp': noisy_imgs,
            'coord': hr_coords,
            'cell': cells,
            'gt': hr_gts,
            'label': labels,
            'basic_subset': basic_subsets,
            'TM_subset': TM_subsets
        }
    return collect_fn


def calc_id_loss(pred, gt, model, id_loss_fn):
    # pred: B C H W
    # gt: B C H W
    model.eval()
    # with torch.no_grad():
    pred_feat = model(pred, is_train=False)
    gt_feat = model(gt, is_train=False)
    id_loss = id_loss_fn(pred_feat, gt_feat)

    return id_loss


def extract_gallery_features(model, gallery):
    imgs = []
    labels = []
    for data in gallery:
        img = data['inp']
        label = data['label']
        imgs.append(img)
        labels.append(label)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    imgs = imgs.cuda()
    labels = labels.cuda()
    
    # torch.set_grad_enabled(False)
    model.eval()
    with torch.no_grad():
        gallery_features = model(imgs, is_train=False)
    return gallery_features, labels

    
class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2):
        cosine_sim = F.cosine_similarity(x1, x2)
        loss = 1 - cosine_sim

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss