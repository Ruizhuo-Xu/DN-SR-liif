CUDA_VISIBLE_DEVICES=3 \
python save_image.py \
    --input /home/rz/data/lock3dface/dataset_1028_new_no_aug_lr_noise0.001 \
    --model save/lock3dface_ori_SR_1_4_edsr-8-noise0.001-v2-id1.0-cos/epoch-best.pth \
    --resolution '128,128' \
    --replace 'sr-v2-id1.0cos'
