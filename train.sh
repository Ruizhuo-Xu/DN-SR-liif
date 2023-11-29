#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python train_liif_v2.py \
    --config configs/train-depthFace/train_edsr-baseline-liif-lock-v2-id.yaml \
    --name LIIF-SR --tag no_idLoss --port $MASTER_PORT \
    --save_path save/baseline
