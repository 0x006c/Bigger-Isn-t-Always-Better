#!/bin/bash

python main_fastmri_new.py \
 --config=configs/ve/celeba_2.py \
 --eval_folder=/srv/local/lg/eval/celeba_2 \
 --mode='train'  \
 --workdir=/srv/local/lg/workdir/celeba_2