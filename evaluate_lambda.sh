#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

for lam in  1.0
do
    python evaluate_lambda.py \
        --model celeba_1 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_2 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_3 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_4 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_4_attention \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam
done
