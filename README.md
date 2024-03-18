# Bigger Isn't Always Better: Towards a General Prior for MRI Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2110.05243-red)](https://arxiv.org/abs/2110.05243)

Contains the Code from ---insert arXiv link---. This repository was modified from [scoreMRI](https://github.com/HJ-harry/score-MRI).

## Usage
We have tested the implementation with Python 3.11, ```requirements.txt``` lists all dependencies. Make sure to set the dataset root in the config files and ```mydata.py```.

### Training
The models are trained using ```train_<model>.sh```, where the model is defind by the corresponding config file ```configs/ve/<model>.py```.

### Evaluation 
To evaluate the models run the evaluation scripts, where ```evaluate_fastmri_*.sh``` implements the sampling algorithm by Chung & Ye, while ```evaluate_jalal_*.sh``` the one by Jalal et al.


