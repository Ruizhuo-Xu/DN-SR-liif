CUDA_VISIBLE_DEVICES=3 \
python save_image.py \
    --input /home/rz/data/lock3dface/dataset_1028_new_no_aug_lr_ \
    --model save/lock3dface_DN_SR_all_tricks/epoch-best.pth \
    --resolution '128,128'
