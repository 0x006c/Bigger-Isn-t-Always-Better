#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1

source activate diff_mri
cd /home/---/score-MRI/diffusion-generalization-mri

python main_fastmri_new.py \
 --config=configs/ve/celeba_3.py \
 --eval_folder=/srv/local/---/eval/celeba_3 \
 --mode='train'  \
 --workdir=/srv/local/---/workdir/celeba_3