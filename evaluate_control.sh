#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1

source activate diff_mri
cd /home/---/score-MRI/diffusion-generalization-mri

for model in unet tv
do
    python evaluate_control.py \
        --model $model \
        --fat_suppression t \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04

    python evaluate_control.py \
        --model $model \
        --fat_suppression t \
        --mask_type gaussian1d \
        --acc_factor 8 \
        --center_fraction 0.04

    python evaluate_control.py \
        --model $model \
        --fat_suppression t \
        --mask_type gaussian2d

    python evaluate_control.py \
        --model $model \
        --fat_suppression t \
        --mask_type radial

    python evaluate_control.py \
        --model $model \
        --fat_suppression t \
        --mask_type poisson \
        --acc_factor 15
done