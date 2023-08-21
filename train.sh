CUDA_VISIBLE_DEVICES=3 \
python train_liif_v2.py \
--config configs/train-depthFace/train_edsr-baseline-liif-lock-v2-id.yaml \
--name lock3dface_ori --tag 'SR_1_4_edsr-8-noise0.001-v2-id0.01'
