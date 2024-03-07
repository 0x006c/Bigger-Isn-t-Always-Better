#!/bin/bash

python evaluate_celeba.py \
    --model celeba_4_attention \
    --N 250 \
    --acc_factor 4 \
    --center_fraction 0.04 \
    --brain t \
    --mask_type gaussian1d

python evaluate_celeba.py \
    --model celeba_4 \
    --N 250 \
    --acc_factor 4 \
    --center_fraction 0.04 \
    --brain t \
    --mask_type gaussian1d

python evaluate_celeba.py \
    --model celeba_3 \
    --N 250 \
    --acc_factor 4 \
    --center_fraction 0.04 \
    --brain t \
    --mask_type gaussian1d

python evaluate_celeba.py \
    --model celeba_2 \
    --N 250 \
    --acc_factor 4 \
    --center_fraction 0.04 \
    --brain t \
    --mask_type gaussian1d

python evaluate_celeba.py \
    --model celeba_1 \
    --N 250 \
    --acc_factor 4 \
    --center_fraction 0.04 \
    --brain t \
    --mask_type gaussian1d