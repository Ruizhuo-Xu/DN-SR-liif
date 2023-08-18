import argparse
import os
from PIL import Image
import pdb

import torch
from torchvision import transforms
import cv2 as cv
import numpy as np

import models
from utils import make_coord, calc_normalMap
from test import batched_predict
from glob import glob
from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    # parser.add_argument('--output', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--replace', default='')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    # find all the png file
    img_paths = glob(os.path.join(args.input, '*', '*', '*.png'))
    for img_path in tqdm(img_paths):
        img_depth = cv.imread(img_path)
        img_normal = calc_normalMap(img_depth[:, :, 0]).astype(np.uint8)
        img = np.concatenate([img_depth, img_normal], axis=-1)
        img = transforms.ToTensor()(img)
        # pdb.set_trace()
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        if 'lr' in img_path:
            save_path = img_path.replace('lr', args.replace)
        else:
            print('invalid path')
            continue
        save_dir = Path(save_path).parent
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(pred).save(save_path)
