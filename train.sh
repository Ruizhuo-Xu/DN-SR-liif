#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=3

python train_liif_v2.py \
    --config configs/train-depthFace/train_edsr-baseline-liif-lock-v2-id.yaml \
    --name LIIF-SR --tag '1.0WeightedL1_1.0idLoss_posff_coordEmb' --port $MASTER_PORT \
    --save_path save/baseline_new
