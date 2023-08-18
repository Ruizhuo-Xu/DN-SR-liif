CUDA_VISIBLE_DEVICES=3 \
python train_liif.py \
--config configs/train-depthFace/train_edsr-baseline-liif-lock-2s.yaml \
--name lock3dface_ori --tag 'SR_1_4_edsr-8-noise0.001-2s'
