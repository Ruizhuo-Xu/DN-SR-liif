CUDA_VISIBLE_DEVICES=3 \
python save_image.py \
    --input /home/rz/data/lock3dface/dataset_1028_new_no_aug_lr_noise0.001 \
    --model save/lock3dface_ori_SR_1_4_edsr-8-noise0.001-2s/epoch-best.pth \
    --resolution '128,128' \
    --replace 'sr-2s'
