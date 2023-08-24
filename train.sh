CUDA_VISIBLE_DEVICES=0 \
python train_liif_v2.py \
--config configs/train-depthFace/train_edsr-baseline-liif-lock-v2-id.yaml \
--name lock3dface_ori --tag 'SR_1_4-id1.0-cos-no-norm'
