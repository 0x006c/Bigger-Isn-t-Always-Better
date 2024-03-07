#!/bin/bash
#SBATCH --partition=icg
#SBATCH --nodelist=nvcluster-node4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mail-user=lukas.glaszner@student.tugraz.at
#SBATCH --mail-type=ALL

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

python evaluate_fastmri.py \
    --model fastmri_knee_4_attention \
    --N 250 \
    --acc_factor 15 \
    --mask_type poisson \
    --fat_suppression t

python evaluate_fastmri.py \
    --model fastmri_knee_4 \
    --N 250 \
    --acc_factor 15 \
    --mask_type poisson \
    --fat_suppression t

python evaluate_fastmri.py \
    --model fastmri_knee_3 \
    --N 250 \
    --acc_factor 15 \
    --mask_type poisson \
    --fat_suppression t

python evaluate_fastmri.py \
    --model fastmri_knee_2 \
    --N 250 \
    --acc_factor 15 \
    --mask_type poisson \
    --fat_suppression t

python evaluate_fastmri.py \
    --model fastmri_knee_1 \
    --N 250 \
    --acc_factor 15 \
    --mask_type poisson \
    --fat_suppression t