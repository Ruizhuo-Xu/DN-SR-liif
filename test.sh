#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python test.py --config configs/test_depthFace/test.yaml \
    --model save/baseline_new/LIIF-SR_1.0WeightedL1_1.0idLoss_posff_coordEmb/epoch-last.pth
