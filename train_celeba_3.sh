#!/bin/bash

python main_fastmri_new.py \
 --config=configs/ve/celeba_3.py \
 --eval_folder=/srv/local/lg/eval/celeba_3 \
 --mode='train'  \
 --workdir=/srv/local/lg/workdir/celeba_3