#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

python main_fastmri_new.py \
 --config=configs/ve/celeba_4_attention.py \
 --eval_folder=/srv/local/lg/eval/celeba_4_attention \
 --mode='train'  \
 --workdir=/srv/local/lg/workdir/celeba_4_attention