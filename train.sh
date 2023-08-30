CUDA_VISIBLE_DEVICES=1 \
python train_liif_v2.py \
--config configs/train-depthFace/train_edsr-baseline-liif-lock-v2-id.yaml \
--name lock3dface_ori --tag 'weighted_l1'
