#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
python test.py --config configs/test_depthFace/test.yaml \
    --gpu 3 \
    --model save/lock3dface_ori_SR_1_8-cos-id1.0/epoch-best.pth