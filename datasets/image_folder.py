import os
import json
from PIL import Image
import random

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv

from datasets import register
from utils import add_noise_Guass, calc_normalMap

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, datafile: str, first_k: int=None, repeat: int=1,
                 cache: str='none', augment=True):
        self.repeat = repeat
        self.cache = cache
        # self.add_noise = add_noise
        self.augment = augment
        if '.txt' in datafile:
            file_paths = np.loadtxt(datafile, dtype=str).tolist()
        elif '.npy' in datafile:
            file_paths = np.load(datafile).tolist()
        else:
            raise "invalid datafile!!"
        if first_k is not None:
            file_paths = file_paths[:first_k]
        self.files = []
        if cache == "none":
            self.files = file_paths
        else:
            for file_path in file_paths:
                if cache == 'bin':
                    raise NotImplementedError

                elif cache == 'in_memory':
                    self.files.append(transforms.ToTensor()(
                        cv.imread(file_path)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = cv.imread(x)
            if len(img.shape) == 3:
                img = img[:, :, 0, None]
            elif len(img.shape) == 2:
                img = img[:, :, None]
            else:
                raise
            # if self.add_noise:
            img_noisy = add_noise_Guass(img, var=0.001)
            if self.augment:
                if random.random() < 0.5:
                    # flip horizontal
                    img = cv.flip(img, 1)
                    img_noisy = cv.flip(img_noisy, 1)
                if random.random() < 0.5:
                    # flip vertical
                    img = cv.flip(img, 0)
                    img_noisy = cv.flip(img_noisy, 0)
            # depth = img_noisy[:, :, 0]
            # normal = calc_normalMap(depth).astype(np.uint8)
            # img = np.concatenate([depth[:, :, None], normal], axis=2)
            # img_noisy = np.concatenate([img_noisy, normal], axis=2)
            # pdb.set_trace()
            # img = img[:,:,0]
            # img = np.stack([img, img, img], axis=2)
            # (6, H, W), (3, H, W)
            return transforms.ToTensor()(img_noisy), transforms.ToTensor()(img)
        elif self.cache == 'bin':
            raise NotImplementedError

        elif self.cache == 'in_memory':
            return x


@register('lock3d-folder')
class Lock3DFolder(Dataset):

    def __init__(self, datafile: str, num2id=None, first_k: int=None, repeat: int=1,
                 cache: str='none', data_norm=False, replace_str=None):
        self.repeat = repeat
        self.cache = cache
        self.replace_str = replace_str
        self.data_norm = data_norm
        if num2id:
            self.num2id = np.load(num2id, allow_pickle=True).item()
        else:
            self.num2id = None
        if '.txt' in datafile:
            file_paths = np.loadtxt(datafile, dtype=str).tolist()
        elif '.npy' in datafile:
            file_paths = np.load(datafile).tolist()
        else:
            raise "invalid datafile!!"
        if first_k is not None:
            file_paths = file_paths[:first_k]
        self.files = []
        if cache == "none":
            self.files = file_paths
        else:
            for file_path in file_paths:
                if cache == 'bin':
                    raise NotImplementedError

                elif cache == 'in_memory':
                    self.files.append(transforms.ToTensor()(
                        cv.imread(file_path, flags=cv.IMREAD_UNCHANGED)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]
        if self.replace_str:
            x = x.replace('1028_new', self.replace_str)
        if self.num2id:
            dir_name = x.split(sep='/')[-2]  # 不同系统可能有差异，注意
            num = int(dir_name[:3])
            label = self.num2id[num]
            label = torch.tensor(label)
            subsets = {'NU':0, 'FE': 1, 'PS':2, 'OC':3, 'TM':4}
            basic_subset = subsets[dir_name.split('_')[-2]]  # //NU FE PS OC //TM
            basic_subset = torch.tensor(basic_subset)
            TM_subset = False
            if num in list(self.num2id.keys())[-169:]:
                TM_subset = True
            TM_subset = torch.tensor(TM_subset)
        else:
            raise NotImplementedError

        if self.cache == 'none':
            img = cv.imread(x)[:, :, 0, None]
            img = transforms.ToTensor()(img)
            if self.data_norm:
                img = (img - 0.5) / 0.5

            return {'inp': img,
                    'label': label,
                    'basic_subset': basic_subset,
                    'TM_subset': TM_subset}
        
        elif self.cache == 'bin':
            raise NotImplementedError

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, add_noise=False, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, add_noise=add_noise, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
