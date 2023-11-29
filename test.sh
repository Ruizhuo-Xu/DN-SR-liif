#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python test.py --config configs/test_depthFace/test.yaml \
    --gpu 0 \
    --model save/LIIF-SR_no_idLoss/epoch-last.pth